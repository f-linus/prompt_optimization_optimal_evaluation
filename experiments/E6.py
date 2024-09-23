import asyncio
import logging
import os

from dotenv import load_dotenv
from tqdm import tqdm

from prompt_optimization.llm.open_ai_compliant_llm_engine import (
    OpenAICompliantLLMEngine,
)
from prompt_optimization.operators.mutation import Mutation
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
from prompt_optimization.utils.plot import plot_optimization_trajectories
from prompt_optimization.utils.prompt import (
    add_embedding_space_representations,
    average_pairwise_cosine_similarity,
    average_pairwise_levenshtein_distance,
    best_unvalidated_prompt,
    prompt_set_union,
)

loaded = load_dotenv(override=True)

se_train, se_test = unfirom_summeval_sample()
rb_train, rb_test = stratified_rewardbench_sample(1, 8)

val_steps = [5, 10]

meta_prompt1 = """{rnd}
Generate a variation of the following prompt:

<PROMPT>
{prompt}
</PROMPT>

The generated prompt should be contain all information and instructions of the original prompt, but worded in a different way.
Important: only change the original prompt a tiny bit!

Provide the new instruction inside <INSTR> tags like this:
<INSTR>...</INSTR>
"""

meta_prompt2 = """{rnd}
Generate a variation of the following prompt:

<PROMPT>
{prompt}
</PROMPT>

The generated prompt should be contain all information and instructions of the original prompt, but worded in a different way.
Important: change as much about the prompt as possible, while keeping the meaning and instructions the same.

Provide the new instruction inside <INSTR> tags like this:
<INSTR>...</INSTR>
"""

se_system = """You are an expert system at evaluating created summaries on the dimensions of coherence, consistency, fluency, and relevance."""

se_instructions = """
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

rb_prompt_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # removed tie option as , and \"[[C]]\ " for a tie
)


logger = logging.getLogger()
logger.setLevel(logging.WARNING)


async def main(n, k):
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
        testset=se_train[:k],
        llm=llama31_70b_eval,
        validation=False,
        system_prompt=se_system,
        instruction_execution_prompt="{input}\n\n{instruction}",
    )
    se_metric = DictSimilarityEvaluator2(conf)

    conf = ExactMatchEvaluatorConfig(
        testset=rb_train[:k],
        llm=llama31_70b_eval,
        validation=False,
        output_extraction_regex=r"\[\[.*\]\]",
        system_prompt=rb_system,
        instruction_execution_prompt="{instruction}\n\n{input}",
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

    prompt_set_se = [Prompt(prompt=se_instructions)]
    prompt_set_rb = [Prompt(prompt=rb_prompt_v2)]

    mutator_se = Mutation(llama31_70b, meta_prompt=meta_prompt1)
    mutator_rb = Mutation(llama31_70b, meta_prompt=meta_prompt1)

    for step in tqdm(range(n)):
        # potentially change meta prompt
        if step == 10:
            mutator_se.meta_prompt = meta_prompt2
            mutator_rb.meta_prompt = meta_prompt2

        # trainset eval
        await asyncio.gather(se_metric(prompt_set_se), rb_metric(prompt_set_rb))

        # mutation
        best_se = sorted(prompt_set_se, key=lambda x: x.mean_score, reverse=True)[0]
        best_rb = sorted(prompt_set_rb, key=lambda x: x.mean_score, reverse=True)[0]

        # validation
        if step in val_steps:
            await asyncio.gather(
                se_val([best_unvalidated_prompt(prompt_set_se)]),
                rb_val([best_unvalidated_prompt(prompt_set_rb)]),
            )

        new_prompts = await asyncio.gather(mutator_se([best_se]), mutator_rb([best_rb]))

        prompt_set_se = prompt_set_union(prompt_set_se, new_prompts[0])
        prompt_set_rb = prompt_set_union(prompt_set_rb, new_prompts[1])

    await asyncio.gather(se_metric(prompt_set_se), rb_metric(prompt_set_rb))

    await asyncio.gather(
        se_val([best_unvalidated_prompt(prompt_set_se)]),
        rb_val([best_unvalidated_prompt(prompt_set_rb)]),
    )

    fig, _ = plot_optimization_trajectories(
        prompt_set_se, figsize=(8, 2), n_top_prompt_display=2
    )

    fig.savefig("f1.png", bbox_inches="tight", dpi=300)

    await add_embedding_space_representations(
        prompt_set_se, os.environ["PERSONAL_OPENAI_KEY"]
    )
    await add_embedding_space_representations(
        prompt_set_rb, os.environ["PERSONAL_OPENAI_KEY"]
    )

    return (
        prompt_set_se,
        prompt_set_rb,
        average_pairwise_cosine_similarity(prompt_set_se),
        average_pairwise_cosine_similarity(prompt_set_rb),
        average_pairwise_levenshtein_distance(prompt_set_se),
        average_pairwise_levenshtein_distance(prompt_set_rb),
    )


if __name__ == "__main__":
    n = 20
    k = 5
    result = asyncio.run(main(n, k))
    print(result)
