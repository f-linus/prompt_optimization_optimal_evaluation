import logging

from tqdm import tqdm

from prompt_optimization.protocols.evaluator import Evaluator
from prompt_optimization.protocols.prompt_optimizer import PromptOptimizer
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import PromptSet
from prompt_optimization.utils.prompt import (
    filter_for_unoptimized,
    prompt_set_union,
)

logger = logging.getLogger(__name__)


class BasicPromptOptimizer(PromptOptimizer):
    def __init__(
        self,
        training_metric: Evaluator,
        validation_metric: Evaluator,
        prune_step: PromptSetModifier,
        optimize_step: PromptSetModifier,
    ):
        self.training_metric = training_metric
        self.validation_metric = validation_metric
        self.prune_step = prune_step
        self.optimize_step = optimize_step

    async def run(
        self, prompt_set: PromptSet, n_steps=5, n_best_validation=3
    ) -> PromptSet:
        for step in tqdm(range(n_steps)):
            logger.info(f"Optimization step {step+1}/{n_steps}")

            prompt_set = await self.training_metric(prompt_set)

            assert all([p.mean_score is not None for p in prompt_set])

            selected_prompts = await self.prune_step(filter_for_unoptimized(prompt_set))
            new_prompts = await self.optimize_step(selected_prompts)

            prompt_set = prompt_set_union(prompt_set, new_prompts)

        # final training metric
        prompt_set = await self.training_metric(prompt_set)

        # select n best prompts to run validation
        logger.info(f"Evaluating {n_best_validation} best prompts on validation set")
        selected_prompts = sorted(
            prompt_set, key=lambda p: (p.mean_score, p.prompt), reverse=True
        )[:n_best_validation]
        await self.validation_metric(selected_prompts)

        return prompt_set
