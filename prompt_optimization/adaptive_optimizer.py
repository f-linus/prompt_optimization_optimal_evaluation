import logging
from time import time
from typing import Callable

from pydantic import BaseModel

from prompt_optimization.operators.pruning.top_k_prompt_pruner import (
    TopKPromptPruner,
)
from prompt_optimization.operators.text_gradients.alternating_text_gradient import (
    AlternatingTextGradient,
    AlternatingTextGradientConfig,
    InteractionExtractionException,
    NewInstructionExtractionException,
)
from prompt_optimization.prompt_eval.exact_match_evaluator import (
    ExactMatchEvaluator,
)
from prompt_optimization.protocols.prompt_optimizer import PromptOptimizer
from prompt_optimization.types.evaluation import Evaluation
from prompt_optimization.types.prompt import PromptSet
from prompt_optimization.utils.profiler import Profiler, ProfilingHandler
from prompt_optimization.utils.prompt import (
    filter_for_unoptimized,
    prompt_set_union,
)

logger = logging.getLogger(__name__)


class AdaptiveOptimizerConfig(BaseModel):
    gradient_config: AlternatingTextGradientConfig
    training_metric: ExactMatchEvaluator

    beam_width_schedule: Callable[[int], int] = lambda step: max([10 - step, 2])
    eval_abort_ratio_schedule: Callable[[int], float] = lambda step: min(
        [0.1 * step + 0.5, 0.9]
    )
    meta_learner_steps: int = 5
    base_learner_steps: int = 5

    class Config:
        arbitrary_types_allowed = True


class AdaptiveOptimizer(PromptOptimizer):
    def __init__(self, config: AdaptiveOptimizerConfig):
        self.config = config
        self.prune = TopKPromptPruner(k=self.config.beam_width_schedule(0))

        self.optimize = AlternatingTextGradient(self.config.gradient_config)
        self.meta_prompt_set = [self.optimize.config.starting_meta_prompt]

    async def run(self, prompt_set: PromptSet):
        profiling_handler = ProfilingHandler(f"adaptive_optimization_{time()}")

        with Profiler(profiling_handler, "initial eval"):
            prompt_set = await self.config.training_metric(prompt_set, sequential=True)

        for meta_step in range(self.config.meta_learner_steps):
            with Profiler(profiling_handler, f"meta-step {meta_step+1}"):
                meta_optimization_context = None
                base_optimization_starting_point = None

                for base_step in range(self.config.base_learner_steps):
                    with Profiler(profiling_handler, f"base-step {base_step+1}"):
                        self.prune.k = self.config.beam_width_schedule(
                            meta_step * self.config.base_learner_steps + base_step
                        )
                        optimization_set = await self.prune(
                            filter_for_unoptimized(prompt_set)
                        )

                        if meta_optimization_context is None:
                            meta_optimization_context = optimization_set[0].__repr__()
                            base_optimization_starting_point = max(
                                [p.mean_score for p in optimization_set]  # type: ignore
                            )

                        try:
                            with Profiler(profiling_handler, "optimization"):
                                optimization_set = await self.optimize(optimization_set)

                            # if succeeds add new prompt to meta_optimization_context
                            meta_optimization_context += (
                                "\n" + optimization_set[-1].__repr__()
                            )
                        except NewInstructionExtractionException:
                            meta_optimization_context += "\n<FAILURE> The optimization failed due to the used interaction not guiding the system to provide an instruction that can be extracted using the defined regex. Consider clarifying in the interaction that the new instruction needs to be included inside corresponding XML tags. </FAILURE>"
                            logger.warning("Optimized prompt could not be extracted.")
                        except InteractionExtractionException:
                            meta_optimization_context += "\n<FAILURE> The interaction could not be extracted from the new meta prompt. (i.e. the generation of the last meta-prompt did fail) </FAILURE>"
                            logger.warning(
                                "Interaction or regex could not be extracted from new meta prompt"
                            )
                            break

                        prompt_set = prompt_set_union(prompt_set, optimization_set)

                        with Profiler(profiling_handler, "evaluation"):
                            prompt_set = await self.config.training_metric(
                                prompt_set,
                                eval_abort_ratio=self.config.eval_abort_ratio_schedule(
                                    meta_step * self.config.base_learner_steps
                                    + base_step
                                ),
                                sequential=True,
                            )

                base_optimization_performance = (
                    max([p.mean_score for p in prompt_set])  # type: ignore
                    - base_optimization_starting_point
                )

                if self.optimize.meta_prompt.evaluations is None:
                    self.optimize.meta_prompt.evaluations = []
                self.optimize.meta_prompt.evaluations.append(
                    Evaluation(
                        score=base_optimization_performance,
                        context=meta_optimization_context,
                    )
                )

                # meta-learner step (only if not last round)
                if meta_step + 1 < self.config.meta_learner_steps:
                    with Profiler(profiling_handler, "updating meta-prompt"):
                        print("Optimizing meta-prompt ...")
                        best_meta_prompt = sorted(
                            self.meta_prompt_set,
                            key=lambda p: (p.mean_score, p.prompt),
                            reverse=True,
                        )[0]
                        new_meta_prompt = await self.optimize.meta_optimization_step(
                            best_meta_prompt, meta_optimization_context
                        )
                        self.meta_prompt_set.append(new_meta_prompt)
                        self.optimize.meta_prompt = new_meta_prompt

        return prompt_set
