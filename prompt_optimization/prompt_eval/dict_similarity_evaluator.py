import asyncio
import json
import logging
import math
import re
from typing import Optional

from pydantic import BaseModel

from prompt_optimization.protocols.llm_engine import LLMEngine
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import PromptSet
from prompt_optimization.types.reference import ReferenceSet
from prompt_optimization.types.reference_evaluation import (
    ReferenceEvaluation,
)
from prompt_optimization.utils.llm import (
    logprobs_chatcompletion_to_expected_number_completion,
)
from prompt_optimization.utils.text import cap_length, remove_line_breaks

SYSTEM_PROMPT = "You are an expert evaluation system. Please as helpful as possible."

INSTRUCTION_EXECUTION_PROMPT = "{input}\n\n{instruction}"
EVALUATION_CONTEXT_TEMPLATE = """
The model was given this input:
<INPUT>
{reference_input}
</INPUT>

And used this instruction:
<INSTRUCTION>
{instruction}
</INSTRUCTION>

Which lead to this output:
<ACTUAL_OUTPUT>
{actual_output}
</ACTUAL_OUTPUT>

This was the reference output:
<REFERENCE_OUTPUT>
{reference_output}
</REFERENCE_OUTPUT>

Now a regular expression was used to extract and compare output dictionaries.
For the actual output that is:
{extracted_actual}

while the extracted reference output is:
{extracted_reference}

Their similarity scores is therefore (1.0=perfect match, 0.0=worst match):
{score} {score_comment}
"""

logger = logging.getLogger(__name__)


class DictSimilarityEvaluatorConfig(BaseModel):
    testset: ReferenceSet
    llm: LLMEngine
    instruction_execution_prompt: str = INSTRUCTION_EXECUTION_PROMPT
    only_evaluate_non_evaluated: bool = True
    logprobs_expectation_temperature: float = 1.0
    validation: bool = False

    class Config:
        arbitrary_types_allowed = True


class DictSimilarityEvaluator(PromptSetModifier):
    def __init__(self, config: DictSimilarityEvaluatorConfig):
        self.config = config

        # check if config has logprobs attribute and if so if it set to True to determine if logs probs are available
        self.log_probs_available = (
            hasattr(self.config.llm, "logprobs") and self.config.llm.logprobs
        )

        self.queries_executed = 0

    async def modify(self, prompt_set: PromptSet) -> PromptSet:
        total_queries = len(prompt_set) * len(self.config.testset)

        def extract_dict(
            text: str, reference_dict: Optional[dict] = None
        ) -> tuple[dict, Optional[str]]:
            try:
                extracted_dict = json.loads(
                    re.search(r"\{.*\}", text, re.DOTALL).group(0)  # type: ignore
                )
            except IndexError:
                logger.info(f"Regex dict extraction failed: {text}")
                return {}, "Could not extract dict with regex."
            except AttributeError:
                logger.info(f"JSON dict extraction failed: {text}")
                return {}, "Could not extract dict with regex."
            except json.JSONDecodeError:
                logger.info(f"JSON dict extraction failed: {text}")
                return {}, "Could not extract dict with regex."

            # checks if reference dict is provided
            if reference_dict:
                # check if all keys in reference dict are also present in extracted dict
                for key in reference_dict.keys():
                    if key not in extracted_dict:
                        logger.info(
                            f"Reference key {key} not found in extracted dict: {extracted_dict}"
                        )
                        return {}, f'Key "{key}" from reference dict not found.'

                # check if all values have the same type
                for key in reference_dict.keys():
                    if type(reference_dict[key]) is not type(extracted_dict[key]):
                        logger.info(
                            f"Reference value type mismatch for key {key}: {extracted_dict}"
                        )
                        return (
                            {},
                            f'Value type mismatch for key "{key}". ( {type(reference_dict[key])} != {type(extracted_dict[key])} )',
                        )

            return extracted_dict, None

        async def process_prompt_reference(
            semaphore: asyncio.Semaphore, prompt, reference
        ):
            async with semaphore:
                llm_context = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": self.config.instruction_execution_prompt.format(
                            input=reference.input, instruction=prompt.prompt
                        ),
                    },
                ]

                try:
                    response = await self.config.llm.non_stream_create(llm_context)
                except Exception as e:
                    logger.error(f"LLM eval failed: {e}")
                    return

                logger.info(
                    f"Prompt: {cap_length(remove_line_breaks(prompt.prompt))}\nReference: {cap_length(remove_line_breaks(reference.output))}\nActual: {cap_length(remove_line_breaks(response))}"
                )

                if self.log_probs_available:
                    messages = logprobs_chatcompletion_to_expected_number_completion(
                        response, self.config.logprobs_expectation_temperature
                    )
                else:
                    messages = response

                for message in messages:
                    # extracting the reference output, a failure of this should be unexpected behaviour
                    extracted_reference, extraction_error = extract_dict(
                        reference.output
                    )
                    if extraction_error:
                        logger.error(
                            f"Could not extract reference dict: {reference.output}"
                        )
                        continue

                    # extraction of the predicted output, a failure here is defined in reference to the
                    # reference dict. A failure here can be expected (i.e. a prompt is just not good)
                    extracted_actual, extraction_error = extract_dict(
                        message, extracted_reference
                    )

                    if extraction_error:
                        score = 0.0
                        score_comment = extraction_error
                    else:
                        avg_deviation = sum(
                            [
                                abs(extracted_reference[key] - extracted_actual[key])
                                for key in extracted_reference.keys()
                            ]
                        ) / len(extracted_reference)
                        score = math.exp(-avg_deviation)
                        score_comment = ""

                    eval_context = EVALUATION_CONTEXT_TEMPLATE.format(
                        reference_input=reference.input,
                        instruction=prompt.prompt,
                        actual_output=message,
                        reference_output=reference.output,
                        extracted_actual=json.dumps(extracted_actual),
                        extracted_reference=json.dumps(extracted_reference),
                        score=round(score, 3),
                        score_comment=score_comment,
                    )

                    eval_obj = ReferenceEvaluation(
                        score=score,
                        context=eval_context,
                        reference=reference,
                        actual=json.dumps(extracted_actual),
                        extracted_reference=extracted_reference,
                        extracted_actual=extracted_actual,
                    )

                    if self.config.validation:
                        prompt.validations.append(eval_obj)
                    else:
                        prompt.evaluations.append(eval_obj)

                    self.queries_executed += 1
                    logger.info(
                        f"Queries finished: {self.queries_executed}/{total_queries} ({round(self.queries_executed/total_queries*100, 2)}%)"
                    )

        async def process_prompt(semaphore: asyncio.Semaphore, prompt):
            if self.config.only_evaluate_non_evaluated and not self.config.validation:
                if prompt.evaluations:
                    return

            if self.config.validation:
                prompt.validations = []
            else:
                prompt.evaluations = []

            tasks = [
                process_prompt_reference(semaphore, prompt, reference)
                for reference in self.config.testset
            ]
            await asyncio.gather(*tasks)

        if self.log_probs_available:
            logger.info(
                "Provided LLM has logprobs enabled. Evaluation will use expectations over numerical integers."
            )

        semaphore = asyncio.Semaphore(3)

        prompt_tasks = [process_prompt(semaphore, prompt) for prompt in prompt_set]

        await asyncio.gather(*prompt_tasks)

        # sort evaluations for each prompt
        for prompt in prompt_set:
            if prompt.evaluations is not None and len(prompt.evaluations) > 0:
                prompt.evaluations = sorted(
                    prompt.evaluations,
                    key=lambda e: e.reference.input,  # type: ignore
                )

        return prompt_set
