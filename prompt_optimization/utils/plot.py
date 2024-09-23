import json
import logging
import math
from textwrap import wrap
from typing import Literal, Optional

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from prompt_optimization.types.prompt import Prompt, PromptSet
from prompt_optimization.utils.misc import dedupe_preserve_order
from prompt_optimization.utils.text import cap_length

logger = logging.getLogger(__name__)


def plot_prompt_dict_performances_against_references(
    prompt_set: list[Prompt], figsize=(24, 9), max_per_row=5, ylim=(0, 5.1)
):
    """
    Generate a comprehensive plot comparing the performance of multiple prompts against reference annotations.

    This function creates a multi-panel figure where each panel represents a single evaluation scenario.
    It visualizes how different prompts perform in comparison to expert annotations across various dimensions.

    Parameters:
    -----------
    prompt_set : list[Prompt]
        A list of Prompt objects, each containing evaluations to be plotted.
    figsize : tuple, optional
        The figure size in inches (width, height). Default is (24, 9).
    max_per_row : int, optional
        The maximum number of subplots per row. Default is 5.
    ylim : tuple, optional
        The y-axis limits for all subplots. Default is (0, 5.1).

    Returns:
    --------
    tuple
        A tuple containing:
        - fig : matplotlib.figure.Figure
            The main figure object.
        - axes : numpy.ndarray
            An array of Axes objects for each subplot.

    Raises:
    -------
    AssertionError
        If the prompts in prompt_set have different numbers of evaluations,
        if any prompt's evaluations lack 'annotations' in their reference context,
        or if the prompts have different sets of references used for evaluation.

    Notes:
    ------
    - Each prompt in prompt_set must have the same number of evaluations and reference set.
    - Each evaluation's reference must have an 'annotations' key in its context.
    - The function plots expert annotation statistics (mean, min, max) as blue error bars.
    - Predictions from each prompt are plotted as red diamond markers, with decreasing opacity
      for subsequent prompts.
    - The plot title includes evaluation scores and a truncated reference input.
    - JSON parsing errors for predictions are logged as warnings.

    Example:
    --------
    >>> prompts = [prompt1, prompt2, prompt3]  # List of Prompt objects
    >>> fig, axes = plot_prompt_dict_performances_against_references(prompts)
    >>> plt.show()
    """

    def json_loads(s):
        try:
            return json.loads(s)
        except Exception:
            logger.warning(f"Could not load JSON: {s}")
            return None

    # first assert that all prompts have the same number of evaluations
    assert len(set([len(p.evaluations) for p in prompt_set])) == 1

    # assert that all prompts have in their evaluatinos references with a context that has the key
    # "annotations"
    assert all(
        [
            all(["annotations" in e.reference.context for e in p.evaluations])
            for p in prompt_set
        ]
    )

    # assert that all prompts have the same set of references used for evaluation
    reference_sets = [
        set([e.reference for e in prompt.evaluations]) for prompt in prompt_set
    ]
    for i in range(len(reference_sets) - 1):
        assert reference_sets[i] == reference_sets[i + 1]

    # estimate general structure of overview plot
    num_plots = len(prompt_set[0].evaluations)
    num_rows = math.ceil(num_plots / max_per_row)
    num_cols = min(num_plots, max_per_row)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes_flat = axes.flatten()

    for i, evaluation in enumerate(prompt_set[0].evaluations):
        ref_repr = evaluation.reference.input[:40] + "..."

        expert_annotations = evaluation.reference.context["annotations"]

        # find relevant evaluations across prompts
        prompt_evals = []
        for prompt in prompt_set:
            prompt_evals.append(
                next(
                    eval
                    for eval in prompt.evaluations
                    if eval.reference == evaluation.reference
                )
            )

        scores = [eval.score for eval in prompt_evals]

        predictions = [eval.extracted_actual for eval in prompt_evals]

        # statistics over expert annotations
        dimensions = list(expert_annotations[0].keys())

        expert_means = {
            dim: np.mean([ref[dim] for ref in expert_annotations]) for dim in dimensions
        }
        expert_mins = {
            dim: np.min([ref[dim] for ref in expert_annotations]) for dim in dimensions
        }
        expert_maxs = {
            dim: np.max([ref[dim] for ref in expert_annotations]) for dim in dimensions
        }

        # plot reference statistics
        yerr_lower = [expert_means[dim] - expert_mins[dim] for dim in dimensions]
        yerr_upper = [expert_maxs[dim] - expert_means[dim] for dim in dimensions]

        x = np.arange(len(dimensions))

        axes_flat[i].errorbar(
            x,
            list(expert_means.values()),
            yerr=[yerr_lower, yerr_upper],
            fmt="o",
            color="blue",
            ecolor="lightblue",
            capsize=3,
            capthick=2,
            elinewidth=1,
            label="Reference Range",
            markersize=8,
            zorder=1,
        )

        # plot predictions
        for j, pred in enumerate(predictions):
            if pred == {}:
                continue

            axes_flat[i].scatter(
                x,
                list(pred.values()),
                marker="d",
                color="red",
                label="Actual",
                s=45,
                zorder=2,
                alpha=1.0 - 0.7 * (j / (len(predictions) - 0.9)),
            )

        # title
        scores_repr = ", ".join([str(round(s, 2)) for s in scores])
        axes_flat[i].set_title(f"Scores: {scores_repr}\n{ref_repr}", fontsize=10)

        # cosmetics
        if ylim:
            axes_flat[i].set_ylim(ylim)

        axes_flat[i].set_xticks(x)
        axes_flat[i].set_xticklabels(dimensions)

        axes_flat[i].grid(True, linestyle="--", alpha=0.7, zorder=0)

        axes_flat[i].spines["top"].set_visible(False)
        axes_flat[i].spines["right"].set_visible(False)
        axes_flat[i].spines["left"].set_linewidth(0.5)
        axes_flat[i].spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()
    return fig, axes


def plot_optimization_trajectories(
    prompt_set: list[Prompt],
    figsize: tuple[int, int] = (12, 3.4),
    n_top_prompt_display: int = 1,
    prompt_display_length: int = 230,
    legend: bool = True,
    color_by: Literal["meta_prompt", "starting_prompt", None] = None,
    score_type: Literal["mean_score", "mean_validation_score"] = "mean_score",
    base_line_color="black",
):
    fig, ax = plt.subplots(figsize=figsize)

    def get_score(prompt: Prompt) -> float:
        return getattr(prompt, score_type)

    def get_std(prompt: Prompt) -> float:
        if score_type == "mean_score":
            return prompt.score_std
        elif score_type == "mean_validation_score":
            return prompt.validation_score_std

    best_prompts = set(
        sorted(prompt_set, key=lambda p: get_score(p), reverse=True)[
            :n_top_prompt_display
        ]
    )

    if color_by == "meta_prompt":
        meta_prompts = dedupe_preserve_order([p.meta_prompt_used for p in prompt_set])
        color_dict = {p: plt.cm.Set1(i) for i, p in enumerate(meta_prompts)}
    elif color_by == "starting_prompt":
        starting_prompts = dedupe_preserve_order(
            [p.trajectory[0][1][0] for p in prompt_set]
        )
        color_dict = {p.prompt: plt.cm.Set1(i) for i, p in enumerate(starting_prompts)}
    else:
        color_dict = {}

    legend_elements = []
    legend_elements_annot = []
    used_colors = set()

    i = 0

    for prompt in prompt_set:
        if not prompt.predecessors:
            continue

        for pred in prompt.predecessors:
            color = base_line_color
            marker_color = base_line_color
            marker = "o"
            marker_size = 2
            zorder = 1

            if color_by:
                color_key = (
                    prompt.meta_prompt_used
                    if color_by == "meta_prompt"
                    else prompt.trajectory[0][1][0]
                )
                color = color_dict[color_key]
                marker_color = color
                zorder = 1

                if color_key not in used_colors:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            lw=2,
                            label="\n".join(
                                wrap(
                                    cap_length(color_key, prompt_display_length),
                                    60,
                                )
                            ),
                        )
                    )
                    used_colors.add(color_key)

            if prompt in best_prompts:
                marker = "d"
                marker_color = "red"
                marker_size = 4
                zorder = 2

                i += 1

                # annotate marker with index
                ax.annotate(
                    str(i),
                    (prompt.generation, get_score(prompt)),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

                legend_elements_annot.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        markersize=marker_size,
                        color="none",
                        markerfacecolor=marker_color,
                        markeredgecolor=marker_color,
                        lw=2,
                        label=f"{i}: "
                        + "\n".join(
                            wrap(cap_length(prompt.prompt, prompt_display_length), 60)
                        )
                        + f" Score: {get_score(prompt):.3f} Std: {get_std(prompt):.3f}",
                    )
                )

            # Plot the line and first marker in light grey
            ax.plot(
                [pred.generation, prompt.generation],
                [get_score(pred), get_score(prompt)],
                color=color,
                linewidth=0.5,
                zorder=zorder,
            )
            ax.plot(
                pred.generation,
                get_score(pred),
                marker="o",
                color=color,
                markersize=2,
                zorder=zorder,
            )

            # Plot the second marker with color and increased size
            ax.plot(
                prompt.generation,
                get_score(prompt),
                marker=marker,
                color=marker_color,
                markersize=marker_size,
                zorder=zorder + 1,
            )

    if legend:
        ax.legend(
            handles=legend_elements + legend_elements_annot,
            prop={"size": 8},
            bbox_to_anchor=(1, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel(score_type.replace("_", " ").title())
    ax.grid(True, linestyle="--", alpha=0.7, zorder=0)

    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    return fig, ax


def plot_prompt_embeddings_pca(
    prompt_set: PromptSet,
    highlight_set: Optional[PromptSet] = None,
    marker_size=8,
    figsize=(6.5, 4.5),
):
    assert all([p.embedding is not None for p in prompt_set])

    fig, ax = plt.subplots(figsize=figsize)

    embeddings = np.array([p.embedding for p in prompt_set])
    scores = np.array([p.mean_score for p in prompt_set]).astype(float)
    generations = np.array([p.generation for p in prompt_set])

    pca = PCA(n_components=2)
    prompt_embeddings_pca = pca.fit_transform(embeddings)

    x = prompt_embeddings_pca[:, 0]
    y = prompt_embeddings_pca[:, 1]

    # Create scatter plot
    cmap = plt.get_cmap("viridis")
    cmap.set_bad("grey")

    scatter = ax.scatter(
        x, y, c=scores, s=marker_size, cmap=cmap, plotnonfinite=True, zorder=2
    )

    # Annotate each point with its generation
    for i, txt in enumerate(generations):
        ax.annotate(
            str(txt),
            (x[i], y[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="red",
            zorder=4,
        )

    # Create colorbar without edges
    if not all(np.isnan(scores)):
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.outline.set_visible(False)

    # Highlight
    if highlight_set is not None:
        highlight_indices = [i for i, p in enumerate(prompt_set) if p in highlight_set]
        ax.scatter(
            x[highlight_indices],
            y[highlight_indices],
            color="red",
            s=marker_size * 3,
            zorder=3,
            marker="X",
        )

    # Plot trajectory relations with arrows
    for prompt in prompt_set:
        if prompt.predecessors is None:
            continue

        for pred in prompt.predecessors:
            pred_idx = prompt_set.index(pred)
            prompt_idx = prompt_set.index(prompt)

            dx = x[prompt_idx] - x[pred_idx]
            dy = y[prompt_idx] - y[pred_idx]

            arrow = ax.arrow(
                x[pred_idx],
                y[pred_idx],
                dx,
                dy,
                color="black",
                width=0.0001,
                head_width=0.005,
                head_length=0.015,
                alpha=1,
                zorder=1,
                length_includes_head=True,
            )
            ax.add_patch(arrow)

    # Cosmetics
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.7, zorder=0)

    plt.tight_layout()
    return fig, ax
