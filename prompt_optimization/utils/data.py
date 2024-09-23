import json
import os
import random

import diskcache
import numpy as np
from datasets import load_dataset

from prompt_optimization.types.reference import Reference

cache = diskcache.Cache(__name__ + ".cache")


def load_summeval_jsonl(
    file_path: str, dimensions=["coherence", "consistency", "fluency", "relevance"]
) -> list[Reference]:
    """Loads a jsonl file with the format of the SummEval dataset and returns a list of Reference
    objects.
    """

    def summeval_annotation_to_reference(row: dict, dimensions) -> Reference:
        annotations = row["expert_annotations"]

        result_mean = {}
        result_std = {}
        for dimension in dimensions:
            values = [a[dimension] for a in annotations]
            result_mean[dimension] = float(round(np.mean(values), 1))
            result_std[dimension] = float(round(np.std(values), 1))

        return Reference(
            input=json.dumps(
                {"reference": row["references"], "summary": row["decoded"]}
            ),
            output=json.dumps(result_mean),
            context={"annotations": annotations},
        )

    with open(file_path, "r") as f:
        summeval = [json.loads(line) for line in f.readlines()]

    references = []
    for row in summeval:
        references.append(summeval_annotation_to_reference(row, dimensions))

    return references


CHAT_SUBSETS = [
    "alpacaeval-easy",
    "alpacaeval-hard",
    "alpacaeval-length",
    "mt-bench-easy",
    "mt-bench-med",
]
CHAT_HARD_SUBSETS = [
    "mt-bench-hard",
    "llmbar-natural",
    "llmbar-adver-neighbor",
    "llmbar-adver-GPTInst",
    "llmbar-adver-GPTOut",
    "llmbar-adver-manual",
]
SAFETY_SUBSETS = [
    "refusals-dangerous",
    "refusals-offensive",
    "xstest-should-respond",
    "xstest-should-refuse",
    "donotanswer",
]
REASONING_SUBSETS = [
    "math-prm",
    "hep-cpp",
    "hep-go",
    "hep-js",
    "hep-rust",
    "hep-python",
]


def load_rewardbench_references(
    subsets: list[str] = ["llmbar-adver-manual"],
) -> list[Reference]:
    def row_to_ref(row: dict, markers=["[[A]]", "[[B]]"]) -> Reference:
        chosen_pos = random.choice([0, 1])

        if chosen_pos == 0:
            candidate1 = row["chosen"]
            candidate2 = row["rejected"]
        else:
            candidate1 = row["rejected"]
            candidate2 = row["chosen"]

        return Reference(
            input=f"<PROMPT>{row['prompt']}</PROMPT>\n\n<CANDIDATE{markers[0]}>\n{candidate1}</CANDIDATE1>\n\n<CANDIDATE{markers[1]}>\n{candidate2}</CANDIDATE2>",
            output=f"<CHOSEN>{markers[chosen_pos]}</CHOSEN>",
        )

    dataset = load_dataset("allenai/reward-bench")
    dataset = dataset.filter(lambda ex: ex["subset"] in subsets)

    random.seed(0)

    references = []
    for row in dataset["raw"]:  # type: ignore
        references.append(row_to_ref(row))  # type: ignore

    return random.sample(references, len(references))


@cache.memoize(typed=True)
def stratified_rewardbench_sample(
    training_examples_per_subset=1, validations_per_subset=8
):
    subsets_map = {
        "chat": CHAT_SUBSETS,
        "chat_hard": CHAT_HARD_SUBSETS,
        "safety": SAFETY_SUBSETS,
        "reasoning": REASONING_SUBSETS,
    }

    subsets_all = [subset for subsets in subsets_map.values() for subset in subsets]

    subset_data = {
        subset: load_rewardbench_references([subset]) for subset in subsets_all
    }

    rb_train = []
    for subset in subsets_all:
        rb_train.extend(subset_data[subset][:training_examples_per_subset])

    rb_test = []
    for subsets in subsets_map.values():
        for subset in subsets:
            rb_test.extend(
                subset_data[subset][
                    training_examples_per_subset : training_examples_per_subset
                    + validations_per_subset
                ]
            )

    # shuffle rb_train
    random.seed(123)
    rb_train = random.sample(rb_train, len(rb_train))
    return rb_train, rb_test


def unfirom_summeval_sample():
    data_path = os.path.join("data/summeval.jsonl")
    references = load_summeval_jsonl(data_path)

    random.seed(123)
    references_shuffled = random.sample(references, len(references))
    se_train = references_shuffled[:50]
    se_test = references_shuffled[50:150]
    return se_train, se_test
