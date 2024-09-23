import itertools
from enum import Enum, auto

import numpy as np
from scipy.special import comb


class LLMCallType(Enum):
    EVAL = auto()
    SAMPLE = auto()


class LLMCall:
    def __init__(self, call_type: LLMCallType, prompt_idx: int = None):
        self.call_type = call_type
        self.prompt_idx = prompt_idx

    def __repr__(self) -> str:
        if self.prompt_idx is not None:
            return f"EVAL-P{self.prompt_idx}"
        else:
            return "SAMPLE"


def outcome_matrix_fn(n_values):
    ranges = [np.arange(n + 1) for n in n_values]
    grid = np.meshgrid(*ranges, indexing="ij")

    return np.stack(grid, axis=-1).reshape(-1, len(n_values))


def future_picked_prompt_belief_expec(prompts: np.array, n_evals: np.array):
    # create binomial distributions for each prompt over the number of positive evals
    alphas = prompts[:, 0]
    betas = prompts[:, 1]

    success_probs = alphas / prompts.sum(axis=1)

    # generate all possible combinations of evaluation outcomes
    outcome_matrix = outcome_matrix_fn(n_evals)

    bin_coefficient_matrix = comb(n_evals, outcome_matrix)
    binom_pmfs = (
        bin_coefficient_matrix
        * success_probs**outcome_matrix
        * (1 - success_probs) ** (n_evals - outcome_matrix)
    )
    outcome_probs = binom_pmfs.prod(axis=1)

    # compute posteriors
    new_alphas = alphas + outcome_matrix
    new_betas = betas + (n_evals - outcome_matrix)
    posterior_means = new_alphas / (new_alphas + new_betas)

    picked_posteriors = posterior_means.max(axis=1)

    expected_picked_posterior_mean = (picked_posteriors * outcome_probs).sum()
    return expected_picked_posterior_mean


def trajectory_expected_performance(
    prompts,
    trajectory: list[LLMCall],
    expected_improvement=0.15,
    prior_alpha=1.9,
    prior_beta=3.4,
):
    n_evals = np.array([0] * len(prompts))
    for action in trajectory:
        if action.call_type == LLMCallType.EVAL:
            n_evals[action.prompt_idx] += 1
        elif action.call_type == LLMCallType.SAMPLE:
            # at this point we need to compute the distribution over
            # the performance of the prompt
            # we will pick to sample the new prompt
            belief_over_sample_base = future_picked_prompt_belief_expec(
                prompts, n_evals
            )

            base_prior_mean = prior_alpha / (prior_alpha + prior_beta)
            delta = belief_over_sample_base - base_prior_mean

            alpha = prior_alpha + delta * expected_improvement
            beta = prior_beta + (prior_alpha - alpha)

            prompts = np.append(prompts, np.array([[alpha, beta]]), axis=0)
            n_evals = np.append(n_evals, 0)

    return prompts, future_picked_prompt_belief_expec(prompts, n_evals)


def flatten(xss):
    return [x for xs in xss for x in xs]


def create_exhaustive_trajectories(starting_set_size: int, tree_depth: int):
    trajectories = []

    # every possible way of performing or not performing new prompt samplings
    sampling_configs = itertools.product(*[[False, True]] * tree_depth)

    # for every sampling configuration we need to check every possible way to
    # perform evaluations between the samples
    for sampling_config in sampling_configs:
        avail_prompts = starting_set_size
        current_window_size = 0
        trajectory_components = []
        for i in range(len(sampling_config)):
            if sampling_config[i]:
                avail_prompts += 1
                trajectory_components.append([(LLMCall(LLMCallType.SAMPLE),)])
            else:
                current_window_size += 1
                if i + 1 >= len(sampling_config) or sampling_config[i + 1]:
                    trajectory_components.append(
                        itertools.combinations_with_replacement(
                            [
                                LLMCall(LLMCallType.EVAL, i)
                                for i in range(avail_prompts)
                            ],
                            current_window_size,
                        )
                    )
                    current_window_size = 0

        # combinations across all options for each component
        sampling_config_trajectories = itertools.product(*trajectory_components)
        trajectories += sampling_config_trajectories

    # flatten each component combination into trajectory
    for i in range(len(trajectories)):
        trajectories[i] = flatten(trajectories[i])
    return trajectories


def optimal_action(prompts, search_depth=5, expected_improvement=0.14):
    trajectories = create_exhaustive_trajectories(len(prompts), search_depth)

    best_traj = (None, 0)
    for traj in trajectories:
        _, final_performance_belief = trajectory_expected_performance(
            prompts, traj, expected_improvement=expected_improvement
        )

        if final_performance_belief > best_traj[1]:
            best_traj = (traj, final_performance_belief)

    return best_traj[0][0]
