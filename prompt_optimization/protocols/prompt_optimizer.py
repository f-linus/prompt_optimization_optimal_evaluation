from typing import Protocol

from prompt_optimization.types.prompt import PromptSet


class PromptOptimizer(Protocol):
    async def run(self, prompt_set: PromptSet) -> PromptSet:
        """Runs the prompt optimization process, starting from the initial prompts and iterating
        through evaluation, pruning and optimization steps until some stopping criterion is met.
        """
        ...
