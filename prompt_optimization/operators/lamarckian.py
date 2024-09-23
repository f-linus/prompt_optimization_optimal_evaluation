import asyncio
import logging
import random
import re

from pydantic import BaseModel, Field

from prompt_optimization.protocols.llm_engine import LLMEngine
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import Prompt, PromptSet
from prompt_optimization.types.reference import ReferenceSet

META_PROMPT = """Given following input/output pairs, infer the most likely used instruction.

{input_output_pairs}

Now wrap the most likely instruction in <INSTR>...</INSTR>
Most likely instruction:
"""

logger = logging.getLogger(__name__)


class LamarckianConfig(BaseModel):
    trainset: ReferenceSet
    llm: LLMEngine

    meta_prompt: str = META_PROMPT

    n_references_per_sample: int = Field(1, ge=1)
    n_samples: int = Field(1, ge=1)

    class Config:
        arbitrary_types_allowed = True


class Lamarckian(PromptSetModifier):
    def __init__(self, config: LamarckianConfig):
        self.config = config

    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        random.seed(0)

        queries = []
        for _ in range(self.config.n_samples):
            sample = random.sample(
                self.config.trainset, self.config.n_references_per_sample
            )

            input_output_pairs_string = "\n\n".join(
                [f"Input: {ref.input}\nOutput: {ref.output}" for ref in sample]
            )

            llm_context = [
                {
                    "role": "system",
                    "content": self.config.meta_prompt.format(
                        input_output_pairs=input_output_pairs_string
                    ),
                }
            ]

            queries.append(self.config.llm.non_stream_create(llm_context))

        logger.info(
            f"Lamarckian operator queries: {len(queries)} (each with n={self.config.llm.config.n})"
        )

        responses = await asyncio.gather(*queries)
        for response in responses:
            for completion in response:
                match = re.search(r"<INSTR>(.*)</INSTR>", completion.message.content)

                if match is None:
                    logger.warning(
                        f"Could not find instruction in response: {completion.message.content}"
                    )
                    continue

                instruction = match.group(1)
                if instruction != "":
                    new_prompt = Prompt(prompt=instruction)
                    if new_prompt not in prompt_set:
                        prompt_set.append(new_prompt)
                        logger.info(f"New instruction added: {instruction}")
                else:
                    logger.warning(
                        f"New instruction is empty: {completion.message.content}"
                    )

        return prompt_set
