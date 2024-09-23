import logging
import re

from pydantic import BaseModel
from tqdm.asyncio import tqdm

from prompt_optimization.protocols.evaluator import Evaluator
from prompt_optimization.protocols.llm_engine import LLMEngine
from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)
from prompt_optimization.types.prompt import PromptSet
from prompt_optimization.types.reference import ReferenceSet
from prompt_optimization.types.reference_evaluation import (
    ReferenceEvaluation,
)
from prompt_optimization.utils.prompt import (
    rank_references_by_difficulty,
)

SYSTEM_PROMPT = "You are an expert evaluation system. Please as helpful as possible."

INSTRUCTION_EXECUTION_PROMPT = "{input}\n\n{instruction}"
EVALUATION_CONTEXT_TEMPLATE = """Input:
<INPUT>
{reference_input}
</INPUT>

The model now used this instruction:
<INSTRUCTION>
{instruction}
</INSTRUCTION>

Which lead to this output:
<ACTUAL_OUTPUT>
{actual_output}
</ACTUAL_OUTPUT>

Reference output:
<REFERENCE_OUTPUT>
{reference_output}
</REFERENCE_OUTPUT>

Now regular expression <REGEX>{regex}</REGEX> was used to extract and compare results.
For the actual output that is {extracted_actual}
while the extracted reference output is {extracted_reference}

The score is therefore {score}"""

logger = logging.getLogger(__name__)


class ExactMatchEvaluatorConfig(BaseModel):
    testset: ReferenceSet
    llm: LLMEngine
    output_extraction_regex: str
    instruction_execution_prompt: str = INSTRUCTION_EXECUTION_PROMPT
    system_prompt: str = SYSTEM_PROMPT
    only_evaluate_non_evaluated: bool = True
    validation: bool = False

    class Config:
        arbitrary_types_allowed = True


class ExactMatchEvaluator(Evaluator):
    """Evaluates an instruction on a test set by performing the instruction on all inputs defiend
    in the test set and comparing the output to the expected output.

    Both expected and actual output are parsed through a regular expression to extract the piece
    of information to evaluate the instruction on (e.g. a number after mathematical reasoning).
    """

    def __init__(self, config: ExactMatchEvaluatorConfig):
        self.config = config

    async def modify(
        self, prompt_set: PromptSet, eval_abort_ratio=None, sequential=False
    ) -> PromptSet:
        async def process_prompt_reference(prompt, reference):
            llm_context = [
                {
                    "role": "system",
                    "content": self.config.system_prompt,
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
                logger.error(
                    f"Failed to evaluate prompt: {prompt.__repr__()} with reference: {reference.input}: {e}"
                )
                logger.error("Removed prompt from pool.")
                prompt_set.remove(prompt)
                return

            logger.info(
                f"Prompt: {prompt.prompt}\nReference: {reference.output}\nActual: {response}"
            )

            for completion in response:
                if completion is None:
                    continue

                # extract value from reference
                # here we assume the reference is well-formed
                extraction_match = re.search(
                    self.config.output_extraction_regex, reference.output
                )

                if extraction_match is None:
                    raise ValueError(
                        f"Could not extract value from reference output: {reference.output}"
                    )

                extracted_reference = extraction_match.group(0)

                extracted_actual = None

                extraction_match = re.search(
                    self.config.output_extraction_regex, completion
                )
                if extraction_match is None:
                    logger.info(
                        f"Could not extract value from actual output: {completion}"
                    )
                else:
                    extracted_actual = extraction_match.group(0)

                score = float(extracted_reference == extracted_actual)

                eval_context = EVALUATION_CONTEXT_TEMPLATE.format(
                    reference_input=reference.input,
                    instruction=prompt.prompt,
                    actual_output=completion,
                    reference_output=reference.output,
                    regex=self.config.output_extraction_regex,
                    extracted_actual=extracted_actual,
                    extracted_reference=extracted_reference,
                    score=score,
                )

                eval_obj = ReferenceEvaluation(
                    score=score,
                    context=eval_context,
                    reference=reference,
                    actual=completion,
                    extracted_reference=extracted_reference,
                    extracted_actual=str(extracted_actual),
                )

                if self.config.validation:
                    prompt.validations.append(eval_obj)
                else:
                    prompt.evaluations.append(eval_obj)

        async def process_prompt(prompt):
            if self.config.only_evaluate_non_evaluated and not self.config.validation:
                if prompt.evaluations:
                    return

            if self.config.validation:
                prompt.validations = []
            elif not prompt.evaluations:
                prompt.evaluations = []

            # determine the order of descending reference difficulty to get the best early evaluation
            # abort probabilities (only relevant if sequential=True)
            references_to_evaluate = rank_references_by_difficulty(
                self.config.testset, previous_evaluations
            )

            tasks = [
                process_prompt_reference(prompt, reference)
                for reference in references_to_evaluate
            ]

            if sequential:
                for task in tqdm(
                    tasks,
                    desc=f"Eval on prompt: {prompt.prompt[:30]}...",
                    disable=len(tasks) < 1000,
                ):
                    await task

                    # determine if it is time to abort
                    # i.e. we abort if it is not possible anymore to reach a score above the threshold
                    if eval_abort_ratio is not None:
                        n_evals_left = len(tasks) - len(prompt.evaluations)
                        max_score = (
                            n_evals_left + sum(e.score for e in prompt.evaluations)
                        ) / len(tasks)

                        current_top_score = max(
                            [
                                p.mean_score
                                for p in prompt_set
                                if p.mean_score is not None
                            ]
                        )
                        if max_score < current_top_score * eval_abort_ratio:
                            logger.info(
                                f"Aborting evaluation of prompt {prompt.prompt} because it is not possible to reach a score above {current_top_score * eval_abort_ratio}"
                            )
                            break
            else:
                await tqdm.gather(
                    *tasks,
                    desc=f"Eval on prompt: {prompt.prompt[:30]}...",
                    disable=len(tasks) < 10,
                )

        previous_evaluations = []
        for prompt in prompt_set:
            if prompt.evaluations is not None:
                previous_evaluations += prompt.evaluations

        prompt_tasks = [process_prompt(prompt) for prompt in prompt_set]

        await tqdm.gather(
            *prompt_tasks, desc="Eval on prompt set...", disable=len(prompt_set) < 10
        )

        # sort evaluations for each prompt
        for prompt in prompt_set:
            if prompt.evaluations is not None and len(prompt.evaluations) > 0:
                prompt.evaluations = sorted(
                    prompt.evaluations,
                    key=lambda e: e.reference.input,  # type: ignore
                )

        return prompt_set
