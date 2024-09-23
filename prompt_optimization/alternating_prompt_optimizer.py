import logging
from typing import Optional

from tqdm import tqdm

from prompt_optimization.operators.text_gradients.alternating_text_gradient import (
    AlternatingTextGradient,
    InteractionExtractionException,
    NewInstructionExtractionException,
)
from prompt_optimization.protocols.evaluator import Evaluator
from prompt_optimization.protocols.prompt_optimizer import PromptOptimizer
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.evaluation import Evaluation
from prompt_optimization.types.prompt import PromptSet
from prompt_optimization.utils.prompt import (
    filter_for_unoptimized,
    prompt_set_union,
)

logger = logging.getLogger(__name__)


class AlternatingPromptOptimizer(PromptOptimizer):
    def __init__(
        self,
        training_metric: Evaluator,
        validation_metric: Optional[Evaluator],
        prune_step: PromptSetModifier,
        optimize_step: AlternatingTextGradient,
    ):
        self.training_metric = training_metric
        self.validation_metric = validation_metric
        self.prune_step = prune_step
        self.optimize_step = optimize_step

        self.meta_prompt_set = [self.optimize_step.config.starting_meta_prompt]

        assert not self.training_metric.config.validation

    async def run(
        self,
        prompt_set: PromptSet,
        meta_learner_steps=5,
        base_learner_steps=5,
        n_best_validation=3,
    ) -> PromptSet:
        total_prompt_set = prompt_set.copy()

        # evaluate initial prompt set
        await self.training_metric(prompt_set)

        for meta_step in range(meta_learner_steps):
            prompt_set_tmp = [
                sorted(
                    total_prompt_set,
                    key=lambda p: (p.mean_score, p.prompt),
                    reverse=True,
                )[0]
            ]

            meta_optimization_context = prompt_set_tmp[0].__repr__()
            for _ in tqdm(range(base_learner_steps), desc="Base-learner steps"):
                prompt_set_tmp = await self.training_metric(prompt_set_tmp)
                assert all([p.mean_score is not None for p in prompt_set_tmp])

                selected_prompts = await self.prune_step(
                    filter_for_unoptimized(prompt_set_tmp)
                )

                try:
                    await self.optimize_step(selected_prompts)

                    # if succeeds add new prompt to meta_optimization_context
                    meta_optimization_context += "\n"
                except NewInstructionExtractionException:
                    meta_optimization_context += "\n<FAILURE> The optimization failed due to the used interaction not guiding the system to provide an instruction that can be extracted using the defined regex. Consider clarifying in the interaction that the new instruction needs to be included inside corresponding XML tags. </FAILURE>"
                    logger.warning("Optimized prompt could not be extracted.")
                except InteractionExtractionException:
                    logger.warning(
                        "Interaction or regex could not be extracted from new meta prompt"
                    )
                    break

                prompt_set_tmp = prompt_set_union(prompt_set_tmp, selected_prompts)

            await self.training_metric(prompt_set_tmp)
            total_prompt_set = prompt_set_union(total_prompt_set, prompt_set_tmp)

            # assign score to meta-prompt
            assert all([p.mean_score is not None for p in prompt_set_tmp])
            optimization_starting_point = prompt_set_tmp[0].mean_score
            optimization_maximum = max([p.mean_score for p in prompt_set_tmp])  # type: ignore

            if self.optimize_step.meta_prompt.evaluations is None:
                self.optimize_step.meta_prompt.evaluations = []
            self.optimize_step.meta_prompt.evaluations.append(
                Evaluation(
                    score=optimization_maximum - optimization_starting_point,
                    context=meta_optimization_context,
                )
            )

            print(
                f"Meta-learner step {meta_step+1}/{meta_learner_steps} done. Current prompt set size: {len(total_prompt_set)}"
            )
            print(
                f"Optimization delta: {optimization_maximum-optimization_starting_point}"
            )
            print(
                f"Current best prompt: {sorted(total_prompt_set, key=lambda p: (p.mean_score, p.prompt), reverse=True)[0].__repr__()}"
            )

            # meta-learner step (only if not last round)
            if meta_step + 1 < meta_learner_steps:
                print("Optimizing meta-prompt ...")
                best_meta_prompt = sorted(
                    self.meta_prompt_set,
                    key=lambda p: (p.mean_score, p.prompt),
                    reverse=True,
                )[0]
                new_meta_prompt = await self.optimize_step.meta_optimization_step(
                    best_meta_prompt, meta_optimization_context
                )
                self.meta_prompt_set.append(new_meta_prompt)
                self.optimize_step.meta_prompt = new_meta_prompt

        # select n best prompts to run validation
        if self.validation_metric is not None:
            logger.info(
                f"Evaluating {n_best_validation} best prompts on validation set"
            )
            selected_prompts = sorted(
                total_prompt_set, key=lambda p: (p.mean_score, p.prompt), reverse=True
            )[:n_best_validation]
            await self.validation_metric(selected_prompts)

        return total_prompt_set
