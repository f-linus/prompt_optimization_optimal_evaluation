import asyncio
import logging
import re

from pydantic import BaseModel, Field

from prompt_optimization.protocols.llm_engine import LLMEngine
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import Prompt, PromptSet
from prompt_optimization.utils.prompt import prompt_set_union

GRADIENT_ESTIMATION_SYSTEM_PROMPT = """
You are an expert at improving instructions. Be as helpful as possible.
"""

GRADIENT_ESTIMATION_PROMPT = """
A model used an instruction to solve a task.

{eval_context}

How could the used instruction be improved to improve the model's performance on that example?
Consider how currently the actual output differs from the reference output.
The reference output (extracted and raw) should point you to the right direction.

Think step by step. Ask yourself following questions:
- given the Reference output, what is the actual task at hand? Be precise and think hard.
- is the given instruction even aimed at solving that task?
- if no how could an instruction look like?
- if yes how could the instruction be improved?

Summarize in a few bullet points what needs to be changed about the instruction with respect to only the one example in the context (i.e. do not provide a new instruction yet!)
"""

GRADIENT_SUMMATION_SYSTEM_PROMPT = """
You are an expert at improving instructions. Be as helpful as possible.
"""

GRADIENT_SUMMATION_PROMPT = """
A model used an instruction to solve a task:

<USED_INSTR>{instr}</USED_INSTR>

We have the following necessary improvements that we need to apply to the instruction:

{gradients}

Think about how all of those points of critique can be combined. After laying out your thinking on that,
provide a new instruction that addresses that summed critique.

Provide the final instruction in <INSTR> tags, like this:
<INSTR>...</INSTR>
"""

logger = logging.getLogger(__name__)


class SummedTextGradientConfig(BaseModel):
    llm: LLMEngine
    n_worst_performing_references: int = Field(1, ge=1)

    class Config:
        arbitrary_types_allowed = True


class SummedTextGradient(PromptSetModifier):
    def __init__(self, config: SummedTextGradientConfig):
        self.config = config

    async def optimise_prompt(self, prompt: Prompt) -> PromptSet:
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

        # estimate gradients for each of the n worst performing references
        gradient_queries = []
        for eval in evals_worst_performing:
            gradient_queries.append(
                self.config.llm.non_stream_create(
                    [
                        {
                            "role": "system",
                            "content": GRADIENT_ESTIMATION_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": GRADIENT_ESTIMATION_PROMPT.format(
                                eval_context=eval.context
                            ),
                        },
                    ]
                )
            )
        gradient_responses = await asyncio.gather(*gradient_queries)

        # flatten list
        gradient_responses = [
            response for sublist in gradient_responses for response in sublist
        ]

        # gradient summation (or more like summarization)
        # concatenate all gradient responses
        concatenated_gradients = "\n\n".join(gradient_responses)

        # create a query to estimate a new instruction based on concatenated gradients
        new_instruction_response = await self.config.llm.non_stream_create(
            [
                {
                    "role": "system",
                    "content": GRADIENT_SUMMATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": GRADIENT_SUMMATION_PROMPT.format(
                        instr=prompt.prompt, gradients=concatenated_gradients
                    ),
                },
            ]
        )

        new_prompts = []
        for completion in new_instruction_response:
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
                    predecessors=[prompt],
                )

                new_prompts.append(new_prompt)
            else:
                logger.warning(
                    f"New instruction is empty: {completion.message.content}"
                )

        return new_prompts

    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        queries = []
        for prompt in prompt_set:
            queries.append(self.optimise_prompt(prompt))

        opt_result = await asyncio.gather(*queries)
        new_prompts = []
        for new_prompts in opt_result:
            new_prompts.extend(new_prompts)

        # sort new_prompts by prompt.prompt
        new_prompts = sorted(new_prompts, key=lambda x: x.prompt)

        return prompt_set_union(prompt_set, new_prompts)
