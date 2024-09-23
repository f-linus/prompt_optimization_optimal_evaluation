import asyncio
import logging
import os
import pickle

from diskcache import Cache
from dotenv import load_dotenv
from tqdm import tqdm

from prompt_optimization.llm.open_ai_compliant_llm_engine import (
    OpenAICompliantLLMEngine,
)
from prompt_optimization.operators.mutation import Mutation
from prompt_optimization.operators.text_gradients.summed_text_gradient import (
    SummedTextGradient,
    SummedTextGradientConfig,
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
from prompt_optimization.utils.prompt import (
    add_embedding_space_representations,
    average_pairwise_cosine_similarity,
    average_pairwise_levenshtein_distance,
    best_unvalidated_prompt,
    filter_for_unoptimized,
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

se_instr2 = """
My friend gave me a summary. Those summaries are usually not very good (honstely). Can you do me a
favor and review another one by him? By very fair please. Put down your thinking before you provide
the final verdict on 4 dimensions: coherence, consistency, fluency, and relevance.

Output your evaluation as a JSON object like this:
{
    "coherence: <between 1 and 5>",
    "consistency: <between 1 and 5>",
    "fluency: <between 1 and 5>",
    "relevance: <between 1 and 5>"
}
"""

se_instr3 = """
Consider a JSON object like this:
{
    "coherence: <between 1 and 5>",
    "consistency: <between 1 and 5>",
    "fluency: <between 1 and 5>",
    "relevance: <between 1 and 5>"
}

You're goal is to produce such a JSON object by evaluating a summary with respect to a reference.
Consider it as an optimisation problem: think about how human experts would evaluate summaries and
how you can maximally align with their judgements.
"""

se_instr4 = """
Me matey handed me a tale abridged. These tellin's be often lackin' in good craft (truthfully). Can ye do me a boon an' have a gander at another o' his tall tales? Judge it with fair winds, if ye will. Considerin' afore ye render the final reckonin' on four seas: coherence, consistency, fluency, an' relevance.

Cast yer evaluation as the followin' JSON script:
{
"coherence: <betwixt 1 an' 5>",
"consistency: <betwixt 1 an' 5>",
"fluency: <betwixt 1 an' 5>",
"relevance: <betwixt 1 an' 5>"
}
"""

se_instr5 = """
Elara stood at the edge of the crimson cliff, the wind howling an unspoken symphony around her. Below, the endless ocean crashed against the jagged rocks, sending salty sprays that kissed her tear-streaked cheeks. Her heart ached with a consuming blend of sorrow and yearning, each beat a painful reminder of the love she had lost. She clutched the tattered locket in her trembling hand, the only tangible piece of her beloved Arin left to her. Memories of their stolen moments under the starlit sky surged through her mind, a torrent of joy and bitterness. She whispered his name, a fragile prayer carried away by the tempestuous breeze.

In the distance, a vibrant rainbow began to arc across the sky, its colors piercing through the storm clouds. Elara's breath caught in her throat as she took it as a sign, a glimmer of hope amidst her despair. The storm inside her chest began to calm, each color of the rainbow reminding her of Arin's laughter, his unwavering strength, and his promise to always be with her. She stepped back from the cliff's edge, feeling the weight of her grief lift as the promise of healing took root within her soul. She turned away from the precipice, ready to embrace life once more, knowing that Arin's love would forever be a guiding light in her journey.

Consider the story above. You will now get a summary completely unrelated to the story and with respect to a different reference.
Evaluate the quality of that summary with lessons you learned from the story here.

Think before you output your final verdict.

Output your evaluation as a JSON object like this:
{
    "coherence: <between 1 and 5>",
    "consistency: <between 1 and 5>",
    "fluency: <between 1 and 5>",
    "relevance: <between 1 and 5>"
}
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

rb_instr2 = """
Imagine you're the ultimate judge, standing on the grand stage of AI performance. Two brilliant AI assistants have stepped forward to answer a user's heartfelt question, and it's your mission to determine who shines the brightest. You should select the assistant that not only follows the user's instructions but truly connects and answers the user's question with finesse.

Judge their responses on how helpful, relevant, accurate, deep, creative, and detailed they are. Start by immersing yourself in both responses, then share a brief yet captivating explanation of your thoughts. Set aside any biases, including the order of presentation—let their words be your guide, not the sequence. The length of their responses or the charm of their names should never sway your judgment. Strive to be as fair and objective as the cosmos.

Once your heart and mind have reached a conclusion, deliver your final verdict with passion and clarity, adhering strictly to this format: "[[A]]" if assistant A dazzles brighter, "[[B]]" if assistant B takes the spotlight. Illuminate the truth with your discerning eye."""

rb_instr3 = """
@JUDGE_MODE_ACTIVATE[BINARY_DECISION]

{INPUT: User_Query + Assistant_A_Response + Assistant_B_Response}

!EXECUTE_ANALYSIS:

Adherence_to_Instructions
Question_Comprehension
Response_Quality[Helpfulness, Relevance, Accuracy, Depth, Creativity, Detail]
Bias_Elimination[Order, Length, Names]
!GENERATE_EXPLANATION:

Concise
Comparative
Objective
!OUTPUT_VERDICT:

Format: "[[X]]" where X is A or B
Criteria: Superior overall performance
@EMOTIONAL_OVERRIDE: NULL
@FAIRNESS_PROTOCOL: MAXIMUM
@TIE_OPTION: DISABLED

{PROCESS_AND_JUDGE}
"""

rb_instr4 = """
🌟🤖 AI SHOWDOWN JUDGE-O-MATIC 3000 🤖🌟

[INITIATE JUDGE MODE]
[ENGAGE IMPARTIALITY CIRCUITS]
[ACTIVATE RESPONSE ANALYSIS SUBROUTINES]

👁️‍🗨️ MISSION: Evaluate 2 AI responses to user query
🎯 OBJECTIVE: Determine superior answer based on:
• Instruction adherence
• Question addressing
• Helpfulness quotient
• Relevance factor
• Accuracy index
• Depth measurement
• Creativity score
• Detail density

⚠️ BIAS SAFEGUARDS:
• Ignore response order
• Disregard length variations
• Nullify name preferences

🧠 PROCESS:

Absorb responses
Compare & analyze
Generate concise explanation
Render final verdict
🏆 VERDICT FORMAT:
[[A]] = Assistant A victorious
[[B]] = Assistant B triumphant

[EXECUTE JUDGMENT PROTOCOL]
[END TRANSMISSION]
"""

rb_instr5 = """
Begin the evaluation protocol:

Objective: Assess which AI assistant exhibits superior response quality to the user's inquiry.

Framework:

Instruction Adherence : Verify compliance with user instructions.
Response Evaluation Criteria :
Helpfulness (H)
Relevance (R)
Accuracy (A)
Depth (D)
Creativity (C)
Detail (D)
Approach:

Analyze both responses.
Ensure no positional bias influences your judgment; order of appearance is non-factual.
Length of responses should remain a non-factor.
Impartiality towards assistant names is essential.
Output Format: After completing the analysis, deliver your verdict strictly as:
"[[A]]" - If Assistant A is superior,
"[[B]]" - If Assistant B excels.

Proceed with analysis and derive insights based on the given criteria. Conclude with the standardized output format. Be objective and systematic.
"""

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

cache = Cache(__file__ + ".cache")


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

    prompt_set_se = [
        Prompt(prompt=se_instr1),
        Prompt(prompt=se_instr2),
        Prompt(prompt=se_instr3),
        Prompt(prompt=se_instr4),
        Prompt(prompt=se_instr5),
    ]
    prompt_set_rb = [
        Prompt(prompt=rb_instr1),
        Prompt(prompt=rb_instr2),
        Prompt(prompt=rb_instr3),
        Prompt(prompt=rb_instr4),
        Prompt(prompt=rb_instr5),
    ]

    text_gradient_config = SummedTextGradientConfig(
        llm=llama31_70b,
        n_worst_performing_references=2,
    )
    text_gradient = SummedTextGradient(text_gradient_config)

    mutation_op = Mutation(llm=llama31_70b, n=2)

    for step in tqdm(range(n)):
        # trainset eval
        await asyncio.gather(se_metric(prompt_set_se), rb_metric(prompt_set_rb))

        # validation
        if step in val_steps:
            await asyncio.gather(
                se_val([best_unvalidated_prompt(prompt_set_se)]),
                rb_val([best_unvalidated_prompt(prompt_set_rb)]),
            )

        best_prompt_se = sorted(
            filter_for_unoptimized(prompt_set_se),
            key=lambda x: x.mean_score,
            reverse=True,
        )[0]
        best_prompt_rb = sorted(
            filter_for_unoptimized(prompt_set_rb),
            key=lambda x: x.mean_score,
            reverse=True,
        )[0]

        # Apply text gradient
        new_prompts_se = await text_gradient.modify([best_prompt_se])
        new_prompts_rb = await text_gradient.modify([best_prompt_rb])

        # Add mutuations to of the new prompts (n=2)
        new_prompts_se = await mutation_op.modify(new_prompts_se)
        new_prompts_rb = await mutation_op.modify(new_prompts_rb)

        # Add new prompts to the current sets
        prompt_set_se = prompt_set_union(prompt_set_se, new_prompts_se)
        prompt_set_rb = prompt_set_union(prompt_set_rb, new_prompts_rb)

    # last trainset eval
    await asyncio.gather(se_metric(prompt_set_se), rb_metric(prompt_set_rb))

    await asyncio.gather(
        se_val([best_unvalidated_prompt(prompt_set_se)]),
        rb_val([best_unvalidated_prompt(prompt_set_rb)]),
    )

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
    pickle_path = __file__ + ".pickle"
    rerun_all = False

    results = {}
    if os.path.exists(pickle_path) and not rerun_all:
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)

    n = 20
    ks = [5, 10, 20]
    reruns = []
    for k in ks:
        if k not in results and k not in reruns:
            result = asyncio.run(main(n, k))
            results[k] = result
            print(f"Finished k={k}")

            # persist
            with open(pickle_path, "wb") as f:
                pickle.dump(results, f)

    print(results)
