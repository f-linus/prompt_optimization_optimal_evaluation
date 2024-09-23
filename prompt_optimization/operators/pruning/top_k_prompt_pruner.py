import logging

from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import PromptSet

logger = logging.getLogger(__name__)


class TopKPromptPruner(PromptSetModifier):
    def __init__(self, k: int):
        self.k = k

    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        logger.info(f"Removing all but the best {self.k} prompts from prompt set")

        # dont do anything if the prompt set is already smaller than k
        if len(prompt_set) <= self.k:
            return prompt_set

        best_prompts = sorted(
            prompt_set, key=lambda x: (x.mean_score, x.prompt), reverse=True
        )[: self.k]

        new_set = [p for p in prompt_set if p in best_prompts]

        prompt_set = new_set
        return new_set
