from typing import Protocol

from prompt_optimization.types.prompt import PromptSet


class PromptSetModifier(Protocol):
    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        """
        Modifies a set of prompts. This could be an initial population of prompts, a pruning of
        prompts or an optimization step in which new prompts around existing prompts are explored.

        Parameters:
            prompt_set: The set of prompts to modify.

        Returns:
            PromptSet: The modified set of prompts.
        """
        ...

    async def __call__(self, prompt_set: PromptSet, *args, **kwargs) -> PromptSet:
        return await self.modify(prompt_set, *args, **kwargs)
