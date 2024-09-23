import asyncio
import logging
import os
import pickle

import numpy as np
from diskcache import Cache
from dotenv import load_dotenv
from tqdm import tqdm

from prompt_optimization.llm.open_ai_compliant_llm_engine import (
    OpenAICompliantLLMEngine,
)
from prompt_optimization.operators.text_gradients.basic_text_gradient import (
    BasicTextGradient,
    BasicTextGradientConfig,
)
from prompt_optimization.prompt_eval.dict_similarity_evaluator2 import (
    DictSimilarityEvaluator2,
    DictSimilarityEvaluator2Config,
)
from prompt_optimization.prompt_eval.exact_match_evaluator import (
    ExactMatchEvaluator,
    ExactMatchEvaluatorConfig,
)
from prompt_optimization.types.prompt import Prompt
from prompt_optimization.utils.data import (
    stratified_rewardbench_sample,
    unfirom_summeval_sample,
)
from prompt_optimization.utils.llmcall_valuation import (
    LLMCallType,
    optimal_action,
)
from prompt_optimization.utils.prompt import (
    prompt_set_union,
)

loaded = load_dotenv(override=True)

se_train, se_test = unfirom_summeval_sample()
rb_train, rb_test = stratified_rewardbench_sample(1, 8)

val_steps = [5, 10]

se_system = """You are an expert system at evaluating created summaries on the dimensions of coherence, consistency, fluency, and relevance."""

se_instr1 = """
You will evaluate the quality of a given summary, with respect to a reference on four dimensions.

Output your evaluation as a JSON object like this:
{
    "coherence: <float between 1 and 5>",
    "consistency: <float between 1 and 5>",
    "fluency: <float between 1 and 5>",
    "relevance: <float between 1 and 5>"
}

Before you output the JSON dict, lay out your reasoning step by step.
"""


rb_system = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

rb_instr1 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # removed tie option as , and \"[[C]]\ " for a tie
)


gradient_meta_prompt = """
{eval_context}

Given the use of a that specific instruction and its performance, provide a new instruction inside
<INSTR> tags, like this:
<INSTR>...</INSTR>
"""

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

cache = Cache(__file__ + ".cache")


async def main(action_budget):
    api_keys = [
        os.environ["DEEPINFRA_API_KEY_A1"],
        os.environ["DEEPINFRA_API_KEY_A2"],
        os.environ["DEEPINFRA_API_KEY_A3"],
        os.environ["DEEPINFRA_API_KEY_A4"],
    ]
    llama31_70b = OpenAICompliantLLMEngine(
        base_url=os.environ["DEEPINFRA_BASE_URL"],
        api_keys=api_keys,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
    llama31_70b_eval = OpenAICompliantLLMEngine(
        base_url=os.environ["DEEPINFRA_BASE_URL"],
        api_keys=api_keys,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        temperature=0.0,
    )

    conf = DictSimilarityEvaluator2Config(
        testset=se_train,
        llm=llama31_70b_eval,
        validation=False,
        system_prompt=se_system,
        instruction_execution_prompt="{input}\n\n{instruction}",
        only_evaluate_non_evaluated=False,
    )
    se_metric = DictSimilarityEvaluator2(conf)

    conf = ExactMatchEvaluatorConfig(
        testset=rb_train,
        llm=llama31_70b_eval,
        validation=False,
        output_extraction_regex=r"\[\[.*\]\]",
        system_prompt=rb_system,
        instruction_execution_prompt="{instruction}\n\n{input}",
        only_evaluate_non_evaluated=False,
    )
    rb_metric = ExactMatchEvaluator(conf)

    conf = DictSimilarityEvaluator2Config(
        testset=se_test,
        llm=llama31_70b_eval,
        validation=True,
        system_prompt=se_system,
        instruction_execution_prompt="{input}\n\n{instruction}",
    )
    se_val = DictSimilarityEvaluator2(conf)

    conf = ExactMatchEvaluatorConfig(
        testset=rb_test,
        llm=llama31_70b_eval,
        validation=True,
        output_extraction_regex=r"\[\[.*\]\]",
        system_prompt=rb_system,
        instruction_execution_prompt="{instruction}\n\n{input}",
    )
    rb_val = ExactMatchEvaluator(conf)

    prompt_set_se = [
        Prompt(prompt=se_instr1),
    ]
    prompt_set_rb = [
        Prompt(prompt=rb_instr1),
    ]

    text_gradient_config_se = BasicTextGradientConfig(
        llm=llama31_70b,
        n_worst_performing_references=2,
        meta_prompt=gradient_meta_prompt,
    )
    text_gradient_se = BasicTextGradient(text_gradient_config_se)

    text_gradient_config_rb = BasicTextGradientConfig(
        llm=llama31_70b,
        n_worst_performing_references=2,
        meta_prompt=gradient_meta_prompt,
    )
    text_gradient_rb = BasicTextGradient(text_gradient_config_rb)

    beliefs_se = np.array([[1.9, 3.4]])
    beliefs_rb = np.array([[1.9, 3.4]])

    action_sequence_se = []
    action_sequence_rb = []
    for action_idx in tqdm(range(action_budget)):
        action_se = optimal_action(beliefs_se)
        action_rb = optimal_action(beliefs_rb)

        action_sequence_se.append(action_se)
        action_sequence_rb.append(action_rb)

        if action_se.call_type == LLMCallType.EVAL:
            eval_prompt = action_se.prompt_idx

            n_evals = (
                len(prompt_set_se[eval_prompt].evaluations)
                if prompt_set_se[eval_prompt].evaluations
                else 0
            )
            reference_for_eval = se_train[n_evals]

            se_metric.config.testset = [reference_for_eval]
            await se_metric.modify([prompt_set_se[eval_prompt]])

            success = prompt_set_se[eval_prompt].evaluations[-1].score > 0.3

            # update belief for prompt
            beliefs_se[eval_prompt, 0] += int(success)
            beliefs_se[eval_prompt, 1] += 1 - int(success)
        else:
            belief_means = beliefs_se[:, 0] / (beliefs_se[:, 0] + beliefs_se[:, 1])
            best_prompt_se = prompt_set_se[np.argmax(belief_means)]
            best_prompt_belief_mean = belief_means[np.argmax(belief_means)]

            new_prompts_se = await text_gradient_se.modify([best_prompt_se])
            prompt_set_se = prompt_set_union(prompt_set_se, new_prompts_se)

            # add prompt to beliefs
            delta = best_prompt_belief_mean - (1.9 / (1.9 + 3.4))
            alpha = 1.9 + delta * 0.18
            beta = 3.4 + (1.9 - alpha)
            beliefs_se = np.vstack([beliefs_se, [alpha, beta]])

        if action_rb.call_type == LLMCallType.EVAL:
            eval_prompt = action_rb.prompt_idx

            n_evals = (
                len(prompt_set_rb[eval_prompt].evaluations)
                if prompt_set_rb[eval_prompt].evaluations
                else 0
            )
            reference_for_eval = rb_train[n_evals]

            rb_metric.config.testset = [reference_for_eval]
            await rb_metric.modify([prompt_set_rb[eval_prompt]])

            success = prompt_set_rb[eval_prompt].evaluations[-1].score > 0.3

            # update belief for prompt
            beliefs_rb[eval_prompt, 0] += int(success)
            beliefs_rb[eval_prompt, 1] += 1 - int(success)
        else:
            belief_means = beliefs_rb[:, 0] / (beliefs_rb[:, 0] + beliefs_rb[:, 1])
            best_prompt_rb = prompt_set_rb[np.argmax(belief_means)]
            best_prompt_belief_mean = belief_means[np.argmax(belief_means)]

            new_prompts_rb = await text_gradient_rb.modify([best_prompt_rb])
            prompt_set_rb = prompt_set_union(prompt_set_rb, new_prompts_rb)

            # add prompt to beliefs
            delta = best_prompt_belief_mean - (1.9 / (1.9 + 3.4))
            alpha = 1.9 + delta * 0.18
            beta = 3.4 + (1.9 - alpha)
            beliefs_rb = np.vstack([beliefs_rb, [alpha, beta]])

    return (
        prompt_set_se,
        prompt_set_rb,
        action_sequence_se,
        action_sequence_rb,
    )


if __name__ == "__main__":
    pickle_path = __file__ + ".pickle"
    rerun_all = False

    results = None
    if os.path.exists(pickle_path) and not rerun_all:
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)

    api_keys = [
        os.environ["DEEPINFRA_API_KEY_A1"],
        os.environ["DEEPINFRA_API_KEY_A2"],
        os.environ["DEEPINFRA_API_KEY_A3"],
        os.environ["DEEPINFRA_API_KEY_A4"],
    ]
    llama31_70b = OpenAICompliantLLMEngine(
        base_url=os.environ["DEEPINFRA_BASE_URL"],
        api_keys=api_keys,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
    llama31_70b_eval = OpenAICompliantLLMEngine(
        base_url=os.environ["DEEPINFRA_BASE_URL"],
        api_keys=api_keys,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        temperature=0.0,
    )

    conf = DictSimilarityEvaluator2Config(
        testset=se_train,
        llm=llama31_70b_eval,
        validation=False,
        system_prompt=se_system,
        instruction_execution_prompt="{input}\n\n{instruction}",
        only_evaluate_non_evaluated=False,
    )
    se_metric = DictSimilarityEvaluator2(conf)

    conf = ExactMatchEvaluatorConfig(
        testset=rb_train,
        llm=llama31_70b_eval,
        validation=False,
        output_extraction_regex=r"\[\[.*\]\]",
        system_prompt=rb_system,
        instruction_execution_prompt="{instruction}\n\n{input}",
        only_evaluate_non_evaluated=False,
    )
    rb_metric = ExactMatchEvaluator(conf)

    conf = DictSimilarityEvaluator2Config(
        testset=se_test,
        llm=llama31_70b_eval,
        validation=True,
        system_prompt=se_system,
        instruction_execution_prompt="{input}\n\n{instruction}",
    )
    se_val = DictSimilarityEvaluator2(conf)

    conf = ExactMatchEvaluatorConfig(
        testset=rb_test,
        llm=llama31_70b_eval,
        validation=True,
        output_extraction_regex=r"\[\[.*\]\]",
        system_prompt=rb_system,
        instruction_execution_prompt="{instruction}\n\n{input}",
    )
    rb_val = ExactMatchEvaluator(conf)

    best_se = sorted(results[0], key=lambda p: p.mean_score, reverse=True)[:3]
    asyncio.run(se_val(best_se))

    best_rb = sorted(results[1][:-1], key=lambda p: p.mean_score, reverse=True)[:3]
    asyncio.run(rb_val(best_rb))

    # results = asyncio.run(main(85))

    # persist
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)

    print(results)
