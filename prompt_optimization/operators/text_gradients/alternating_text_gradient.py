import asyncio
import json
import logging
import re
from typing import TypeGuard

from pydantic import BaseModel, Field

from prompt_optimization.protocols.llm_engine import LLMEngine
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import Prompt, PromptSet
from prompt_optimization.types.reference_evaluation import (
    ReferenceEvaluation,
)

logger = logging.getLogger(__name__)

STARTING_META_PROMPT = Prompt(
    prompt="""
<INTERACTION>
[
    {
        "role": "system",
        "content": "You are an expert in providing feedback on instructions. You are tasked with helping a user improve the quality of instructions given to a friend for evaluating AI outputs. The user has provided you with an example of an instruction, the output of the friend, the reference output, and the score the friend received for this example. Your task is to provide feedback on the instruction and suggest improvements."
    },
    {
        "role": "user",
        "content": "I gave my friend a specific instruction to evaluate two candidate outputs for a specific prompt. My friend was told to choose which candidate reply he prefers based on the quality of the response. This is the instruction I gave him: '{instruction}'.\\n\\nI now compared his expressed preferences for a few examples with some objective reference I have. Unfortunately, they do ont fully match and I think I have to give him a more accurate/better instruction to do the job. Here is one such example:\\n\\nExample 1:\\n\\nInput: '{example1_input}'\\n\\nOutput: '{example1_output_raw}'\\n\\nExtracted: '{example1_output_extracted}'\\n\\nReference: '{example1_reference_output}'\\n\\nConsidering the mismatch between reference and extracted output, my friend received the score {example1_score} for this example. Considering this example, what aspects of the instruction I gave him need improvement? Give detailled feedback for the instruction and think step by step."
    },
    {
        "role": "assistant",
        "content": "<SOME RESPONSE WE WILL GET>"
    },
    {
        "role": "user",
        "content": "Thanks for the feedback! Choose one of four actions that you think has the highest probability of improving the performance of the instruction:\\n1. Add a sentence somewhere in the instruction\\n2. Remove a sentence\\n3. Change/replace a sentence\\n4. Revert to a previous better-performing instruction\\nChoose the action and provide your reasoning, taking into account how this change relates to previous attempts. Afterwards, specify where in the instruction you would perform the action (or which previous instruction to revert to). Then provide the updated instruction in <INSTRUCTION>..</INSTRUCTION> tags."
    }
]
</INTERACTION>

<EXTRACTION_REGEX>
<INSTRUCTION>(.*)</INSTRUCTION>
</EXTRACTION_REGEX>
"""
)

META_META_SYSTEM_PROMPT = "You are an expert in improving prompt optimization systems."

META_META_INTERACTION = [
    """My best friend created an LLM-based system to optimize LLM instructions.

This is how he designed the system (i.e. the system consists of an interaction-design and a extraction regex):
{meta_prompt}

The system lead to following optimization trajectory:
<OPTIMIZATION_TRAJECTORY>
{trajectory_context}
</OPTIMIZATION_TRAJECTORY>

Now think about how that system could be changed to improve the optimization trajectory.

These are the placeholders you can use when designing the interaction:
# available place holders:
# instruction
# example1_score
# example1_input
# example1_output_raw
# example1_output_extracted
# example1_reference_output
# example2_ ...
# reply1
# reply2 ...
# history: this will be replaced by the history of previous instructions and their scores (in that case you dont need to use example1_... placeholders)

Think step by step and lay out your reasoning. Think particulary hard about the nature of the search step that the current system applies (how big are steps, how are the regularized etc.).

After your thinking, suggest a (small) change to the system (e.g. changing the interaction a bit). Make sure you understand that we have to define an optimal prompt improvement interaction through the "user" parts of the conversation.
""",
    """Perfect. Now consider the current design of the interation and formalise your thinking into a (small) suggested change. Provide a new interaction in the same format as before in <INTERACTION>...</INTERACTION> tags. Only change the meta-prompt a bit and keep in mind that you only change the "user" bits. Also never forget that the model needs to be told that it uses <INSTRUCTION> tags for updated instructions.""",
]

META_META_EXTRACTION_REGEX = r"<INTERACTION>(.*)</INTERACTION>"


def are_all_reference_evaluations(
    evaluations: list,
) -> TypeGuard[list[ReferenceEvaluation]]:
    return all(isinstance(e, ReferenceEvaluation) for e in evaluations)


class InteractionExtractionException(Exception):
    def __init__(self, message):
        self.message = message


class NewInstructionExtractionException(Exception):
    def __init__(self, message):
        self.message = message


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def trajectory_context(prompt: Prompt, history_size=4) -> str:
    history_str = ""
    for generation, (score, prompts) in enumerate(prompt.trajectory[-history_size:]):
        gen_prompt = prompts[0]
        assert gen_prompt.evaluations is not None and len(gen_prompt.evaluations) > 0

        bad_example = sorted(
            gen_prompt.evaluations, key=lambda x: (x.score, x.reference.input)
        )[0]

        history_str += f"<INSTRUCTION{generation+1}>\n{gen_prompt.prompt}\nScore: {score}\n<SUBOPTIMAL_EXAMPLE>\nInput: {bad_example.reference.input}\nOutput produced: {bad_example.extracted_actual}\nOutput expected: {bad_example.extracted_reference}\n</SUBOPTIMAL_EXAMPLE>\n</INSTRUCTION{generation+1}>\n"
    return history_str


class AlternatingTextGradientConfig(BaseModel):
    llm_base_level: LLMEngine
    llm_base_level_last_step: LLMEngine
    llm_meta_level: LLMEngine
    starting_meta_prompt: Prompt = STARTING_META_PROMPT
    meta_meta_system_prompt: str = META_META_SYSTEM_PROMPT
    meta_meta_interaction: list = META_META_INTERACTION
    meta_meta_extraction_regex: str = META_META_EXTRACTION_REGEX
    meta_optimization_max_trajetory_size: int = Field(10, ge=1)
    history_size: int = Field(4, ge=1)

    class Config:
        arbitrary_types_allowed = True


class AlternatingTextGradient(PromptSetModifier):
    def __init__(self, config: AlternatingTextGradientConfig):
        self.config = config
        self.meta_prompt = config.starting_meta_prompt

    async def _optimization_step(self, prompt: Prompt) -> list[Prompt]:
        assert prompt.evaluations is not None and len(prompt.evaluations) > 0

        interaction_str_match = re.search(
            r"<INTERACTION>(.*)</INTERACTION>", self.meta_prompt.prompt, re.DOTALL
        )

        if interaction_str_match is not None:
            try:
                interaction = json.loads(interaction_str_match.group(1))
            except Exception:
                raise InteractionExtractionException(
                    "Could not extract interaction from meta prompt into JSON."
                )
        else:
            raise InteractionExtractionException(
                "Could not extract interaction from meta prompt."
            )

        extraction_regex_match = re.search(
            r"<EXTRACTION_REGEX>(.*)</EXTRACTION_REGEX>",
            self.meta_prompt.prompt,
            re.DOTALL,
        )

        if extraction_regex_match is not None:
            extraction_regex = extraction_regex_match.group(1).strip()
        else:
            raise InteractionExtractionException(
                "Could not extract extraction regex from meta prompt."
            )

        # check if all evaluations are reference evaluations (so that they have the necessary information)
        if not are_all_reference_evaluations(prompt.evaluations):
            raise TypeError("All evaluations must be ReferenceEvaluation instances")

        bad_example = sorted(
            prompt.evaluations, key=lambda x: (x.score, x.reference.input)
        )[0]

        context = SafeDict(
            {
                "instruction": prompt.prompt,
                "example1_score": bad_example.score,
                "example1_input": bad_example.reference.input,
                "example1_output_raw": bad_example.actual,
                "example1_output_extracted": bad_example.extracted_actual,
                "example1_reference_output": bad_example.extracted_reference,
                "history": trajectory_context(
                    prompt, history_size=self.config.history_size
                ),
            }
        )

        llm_context = interaction[:1].copy()
        idx = 1

        while idx + 2 < len(interaction):
            llm_context.append(interaction[idx].copy())

            # fill in the context (if placeholders are present)
            llm_context[-1]["content"] = llm_context[-1]["content"].format_map(context)

            # llm call
            completions = await self.config.llm_base_level.non_stream_create(
                llm_context
            )
            response_content = completions[0]

            # add to LLM context
            llm_context.append({"role": "assistant", "content": response_content})

            # we add two to the index to skip the assistant response place holders
            idx += 2

        # last llm call
        llm_context.append(interaction[idx].copy())
        llm_context[-1]["content"] = llm_context[-1]["content"].format_map(context)

        completions = await self.config.llm_base_level_last_step.non_stream_create(
            llm_context
        )

        new_prompts = []
        for completion in completions:
            # extract updated instruction
            updated_instruction_match = re.search(
                extraction_regex, completion, re.DOTALL
            )

            if updated_instruction_match is None:
                continue

            updated_instruction = updated_instruction_match.group(1).strip()
            new_prompts.append(
                Prompt(
                    prompt=updated_instruction,
                    instantiation_context=json.dumps(llm_context),
                    predecessors=[prompt],
                    meta_prompt_used=self.meta_prompt.prompt,
                )
            )

        if len(new_prompts) == 0:
            raise NewInstructionExtractionException(
                "Could not extract new instruction from response."
            )

        return new_prompts

    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        queries = []
        for prompt in prompt_set:
            queries.append(self._optimization_step(prompt))

        responses = await asyncio.gather(*queries)
        for response in responses:
            for new_prompt in response:
                if new_prompt not in prompt_set:
                    prompt_set.append(new_prompt)
                    logger.info(f"New instruction added: {new_prompt.prompt}")

        return prompt_set

    async def meta_optimization_step(
        self, meta_prompt: Prompt, optimization_context: str
    ) -> Prompt:
        llm_context = [
            {"role": "system", "content": self.config.meta_meta_system_prompt}
        ]
        for instruction in self.config.meta_meta_interaction:
            llm_context.append(
                {
                    "role": "user",
                    "content": instruction.format(
                        meta_prompt=meta_prompt.prompt,
                        trajectory_context=optimization_context,
                    ),
                }
            )

            completions = await self.config.llm_meta_level.non_stream_create(
                llm_context
            )
            completion = completions[0]

            llm_context.append({"role": "assistant", "content": completion})

        mew_meta_instruction_match = re.search(
            self.config.meta_meta_extraction_regex, completion, re.DOTALL
        )

        if mew_meta_instruction_match is None:
            raise InteractionExtractionException(
                "Could not extract meta interaction from response"
            )

        new_meta_interaction = mew_meta_instruction_match.group(1).strip()

        meta_prompt_new = f"""
        <INTERACTION>
        {new_meta_interaction}
        </INTERACTION>

        <EXTRACTION_REGEX>
        <INSTRUCTION>(.*)</INSTRUCTION>
        </EXTRACTION_REGEX>
        """

        # update meta prompt
        return Prompt(
            prompt=meta_prompt_new,
            predecessors=[self.meta_prompt],
            instantiation_context=json.dumps(llm_context),
        )
