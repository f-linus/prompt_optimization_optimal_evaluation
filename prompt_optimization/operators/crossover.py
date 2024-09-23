import logging
import re

from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import Prompt, PromptSet

SYSTEM_PROMPT = "You are an expert combing two prompts into a new prompt."

META_PROMPT = """Create a new prompt based on the two following:

<PROMPT1>
{prompt1}
</PROMPT1>

<PROMPT2>
{prompt2}
</PROMPT2>

First analayse how both prompts differ. Then create a new prompt that combines the two prompts
into a new prompt.

Provide the new instruction inside <INSTR> tags like this:
<INSTR>...</INSTR>
"""

logger = logging.getLogger(__name__)


class Crossover(PromptSetModifier):
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
        assert len(prompt_set) == 2

        llm_context = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.meta_prompt.format(
                    prompt1=prompt_set[0].prompt, prompt2=prompt_set[1].prompt
                ),
            },
        ]

        responses = await self.llm.non_stream_create(llm_context)
        new_prompts = []
        for completion in responses:
            instruction_match = re.search(r"<INSTR>(.*)</INSTR>", completion, re.DOTALL)
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
                    predecessors=prompt_set,
                    meta_prompt_used=self.meta_prompt,
                )

                new_prompts.append(new_prompt)
                logger.info(f"New instruction added: {instruction}")
            else:
                logger.warning(
                    f"New instruction is empty: {completion.message.content}"
                )

        return new_prompts
