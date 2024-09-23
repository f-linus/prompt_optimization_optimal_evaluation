import asyncio
import logging
import re

from pydantic import BaseModel, Field

from prompt_optimization.protocols.llm_engine import LLMEngine
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import Prompt, PromptSet

SYSTEM_PROMPT = (
    "You are an expert at improving instructions. Be as helpful as possible."
)

META_PROMPT = """A model used an instruction to solve a task.

{eval_context}

How could the used instruction be improved to improve the model's performance? Consider how currently
the actual output differs from the reference output. The reference output (extracted and raw) should
point you to the right direction.

Think step by step. Ask yourself following questions:
- given the Reference output, what is the actual task at hand? Be precise and think hard.
- is the given instruction even aimed at solving that task?
- if no how could an instruction look like?
- if yes how could the instruction be improved?

After laying out your thinking on these questions, provide a suggestion inside <INSTR>, like this:
<INSTR>...</INSTR>
"""

logger = logging.getLogger(__name__)


class BasicTextGradientConfig(BaseModel):
    llm: LLMEngine
    meta_prompt: str = META_PROMPT
    n_worst_performing_references: int = Field(1, ge=1)

    class Config:
        arbitrary_types_allowed = True


class BasicTextGradient(PromptSetModifier):
    def __init__(self, config: BasicTextGradientConfig):
        self.config = config

    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        queries = []
        for prompt in prompt_set:
            assert prompt.evaluations is not None, "Prompt has no evaluations"
            assert len(prompt.evaluations) > 0, "Prompt has no evaluations"

            # get the evaluations (text-form) for the n references with the worst prompt performance

            # presort based on context string for deterministic behavior
            # (since .evaluations can be in arbitrary order and scores might be equal for different references)
            eval_presorted = sorted(
                prompt.evaluations, key=lambda x: x.context, reverse=False
            )

            # actual sort based on score
            evals_sorted = sorted(eval_presorted, key=lambda x: x.score, reverse=False)

            # select the n worst performing references
            evals_worst_performing = evals_sorted[
                : self.config.n_worst_performing_references
            ]

            # every relevant (i.e. reference on which the model did not do well) evaluation
            # is now taken to improve the instruction
            combined_eval_context = "\n\n".join(
                [eval.context for eval in evals_worst_performing]
            )
            llm_context = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": self.config.meta_prompt.format(
                        eval_context=combined_eval_context
                    ),
                },
            ]

            queries.append(self.config.llm.non_stream_create(llm_context))

        logger.info(f"Text gradient queries: {len(queries)})")

        # gather responses
        responses = await asyncio.gather(*queries)
        for prompt_idx, response in enumerate(responses):
            for completion in response:
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
                        predecessors=[prompt_set[prompt_idx]],
                        meta_prompt_used=self.config.meta_prompt,
                    )

                    if new_prompt not in prompt_set:
                        prompt_set.append(new_prompt)
                        logger.info(f"New instruction added: {instruction}")
                else:
                    logger.warning(
                        f"New instruction is empty: {completion.message.content}"
                    )

        return prompt_set
