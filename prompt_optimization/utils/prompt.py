import os

import numpy as np
from Levenshtein import distance
from openai import OpenAI

from prompt_optimization.caching.disk_cache_client import DiskCacheClient
from prompt_optimization.types.evaluation import Evaluation
from prompt_optimization.types.prompt import Prompt, PromptSet
from prompt_optimization.types.reference import Reference

cache_client = DiskCacheClient(
    cache_name=os.path.join(os.path.dirname(__file__), "openai_embeddings_cache")
)


def prompt_set_from_strings(prompt_strings: list[str]) -> PromptSet:
    """
    Convert a list of prompt strings to a PromptSet.

    Args:
        prompt_strings (list[str]): A list of prompt strings.

    Returns:
        PromptSet: A list of Prompt objects created from the input strings.
    """

    return [Prompt(prompt=prompt_string) for prompt_string in prompt_strings]


def filter_for_best_prompt_per_generation_per_metaprompt(
    prompt_set: PromptSet,
) -> PromptSet:
    """
    Filter a PromptSet to keep only the best prompt for each generation and meta-prompt combination.

    Args:
        prompt_set (PromptSet): The original set of prompts to filter.

    Returns:
        PromptSet: A filtered set of prompts containing only the highest-scoring prompt
                   for each unique combination of generation and meta-prompt.
    """

    prompt_set_filtered = []

    meta_prompts = set([p.meta_prompt_used for p in prompt_set])
    for mp in meta_prompts:
        promts = [p for p in prompt_set if p.meta_prompt_used == mp]
        generations = set([p.generation for p in promts])

        # only add highest score prompt from each generation
        for gen in generations:
            prompts_gen = [p for p in promts if p.generation == gen]
            best_prompt = max(prompts_gen, key=lambda x: x.mean_score)
            prompt_set_filtered.append(best_prompt)

    return prompt_set_filtered


def filter_for_unoptimized(prompt_set: PromptSet) -> PromptSet:
    """
    Filter a PromptSet to keep only the prompts that have not been optimized.

    Args:
        prompt_set (PromptSet): The original set of prompts to filter.

    Returns:
        PromptSet: A filtered set of prompts containing only the prompts that have not been optimized.
    """

    predecessors_all = set()
    for p in prompt_set:
        if p.predecessors is None:
            continue

        for pred in p.predecessors:
            predecessors_all.add(pred)

    return [p for p in prompt_set if p not in predecessors_all]


def prompt_set_union(prompt_set1: PromptSet, prompt_set2: PromptSet) -> PromptSet:
    """
    Return the union of two PromptSets.

    Args:
        prompt_set1 (PromptSet): The first set of prompts.
        prompt_set2 (PromptSet): The second set of prompts.

    Returns:
        PromptSet: A list of Prompt objects containing all the unique prompts from both input sets.
    """

    for prompt in prompt_set2:
        if prompt not in prompt_set1:
            prompt_set1.append(prompt)

    return prompt_set1


async def add_embedding_space_representations(
    prompt_set: PromptSet, openai_api_key: str
) -> None:
    client = OpenAI(api_key=openai_api_key)

    prompt_strings = [p.prompt for p in prompt_set]

    # check if present in cache
    cache_response = await cache_client.get(prompt_strings)
    if cache_response is not None:
        embeddings_obj = cache_response
    else:
        embeddings_obj = client.embeddings.create(
            model="text-embedding-3-large",
            input=prompt_strings,
            encoding_format="float",
        )
        await cache_client.set(prompt_strings, embeddings_obj)

    prompt_embeddings = [piece.embedding for piece in embeddings_obj.data]

    for i, prompt in enumerate(prompt_set):
        prompt.embedding = prompt_embeddings[i]


def rank_references_by_difficulty(
    references: list[Reference], evaluations: list[Evaluation]
) -> list[Reference]:
    reference_evals = {}
    for evaluation in evaluations:
        if evaluation.reference not in reference_evals:
            reference_evals[evaluation.reference] = []

        reference_evals[evaluation.reference].append(evaluation.score)

    return sorted(
        references,
        key=lambda evaluation: np.mean(reference_evals.get(evaluation, [-1])),
    )


def average_pairwise_cosine_similarity(prompt_set: PromptSet):
    embeddings = np.array([p.embedding for p in prompt_set])

    # Normalize the vectors
    normalized_embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    # Compute the dot product
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Get the upper triangle of the similarity matrix
    upper_tri = np.triu(similarity_matrix, k=1)

    # Calculate the average
    n = embeddings.shape[0]
    average_similarity = np.sum(upper_tri) / (n * (n - 1) / 2)

    return average_similarity


def average_pairwise_levenshtein_distance(prompt_set: PromptSet):
    """Compute the average pairwise Levenshtein distance for a list of strings."""
    strings = [p.prompt for p in prompt_set]

    if not strings:
        return 0

    n = len(strings)
    total_distance = 0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_distance += distance(strings[i], strings[j])
            pair_count += 1

    if pair_count == 0:
        return 0

    return total_distance / pair_count


def best_unvalidated_prompt(prompt_set: PromptSet) -> Prompt:
    return sorted(
        [p for p in prompt_set if p.mean_validation_score is None],
        key=lambda p: p.mean_score,
        reverse=True,
    )[0]
