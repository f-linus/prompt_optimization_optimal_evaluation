import asyncio
import logging
import random
import re

from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import Prompt, PromptSet
from prompt_optimization.utils.prompt import prompt_set_union

SYSTEM_PROMPT = "You are an expert at modifying prompts in different ways while maintaining their semantic meaning."

META_PROMPT = """{rnd}
Generate a variation of the following prompt:

<PROMPT>
{prompt}
</PROMPT>

The generated prompt should be contain all information and instructions of the original prompt, but worded in a different way.

Provide the new instruction inside <INSTR> tags like this:
<INSTR>...</INSTR>
"""

logger = logging.getLogger(__name__)


class Mutation(PromptSetModifier):
    def __init__(
        self,
        llm,
        n=1,
        add_random_number=True,
        system_prompt=SYSTEM_PROMPT,
        meta_prompt=META_PROMPT,
    ):
        self.llm = llm
        self.n = n
        self.add_random_number = add_random_number
        self.system_prompt = system_prompt
        self.meta_prompt = meta_prompt

    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        mutation_queries = []

        async def mutate(self, prompt: Prompt) -> Prompt:
            if self.add_random_number:
                random_number = random.randint(0, 1000)
            else:
                random_number = ""

            llm_context = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.meta_prompt.format(
                        prompt=prompt.prompt, rnd=random_number
                    ),
                },
            ]

            responses = await self.llm.non_stream_create(llm_context)
            new_prompts = []
            for completion in responses:
                instruction_match = re.search(
                    r"<INSTR>(.*)</INSTR>", completion, re.DOTALL
                )
                if instruction_match is None:
                    logger.warning(
                        f"Could not extract new instruction from response: {completion}"
                    )
                    continue

                instruction = instruction_match.group(1).strip()

                if instruction != "":
                    new_prompt = Prompt(
                        prompt=instruction,
                        instantiation_context=completion,
                        predecessors=[prompt],
                        meta_prompt_used=self.meta_prompt,
                    )

                    new_prompts.append(new_prompt)
                    logger.info(f"New instruction added: {instruction}")
                else:
                    logger.warning(
                        f"New instruction is empty: {completion.message.content}"
                    )

            return new_prompts

        for prompt in prompt_set:
            for _ in range(self.n):
                mutation_queries.append(mutate(self, prompt))

        new_prompts = await asyncio.gather(*mutation_queries)

        # flatten list
        new_prompts = [prompt for sublist in new_prompts for prompt in sublist]

        return prompt_set_union(prompt_set, new_prompts)
