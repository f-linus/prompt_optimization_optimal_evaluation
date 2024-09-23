from typing import Optional

import numpy as np
from pydantic import BaseModel

from prompt_optimization.types.evaluation import Evaluation
from prompt_optimization.utils import text


class Prompt(BaseModel):
    """
    Represents a prompt with an associated set of scores, referring to the performance of associated
    completions, potentiall in comparison to some set of references.
    """

    prompt: str
    instantiation_context: str = ""
    evaluations: Optional[list[Evaluation]] = None
    validations: Optional[list[Evaluation]] = None
    predecessors: Optional[list["Prompt"]] = None
    meta_prompt_used: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    @property
    def mean_score(self) -> Optional[float]:
        if self.evaluations is None or len(self.evaluations) == 0:
            return None

        return sum([eval.score for eval in self.evaluations]) / len(self.evaluations)

    @property
    def mean_validation_score(self) -> Optional[float]:
        if self.validations is None or len(self.validations) == 0:
            return None

        return sum([eval.score for eval in self.validations]) / len(self.validations)

    @property
    def score_std(self) -> Optional[float]:
        if self.evaluations is None or len(self.evaluations) == 0:
            return None

        if self.mean_score is None:
            return None

        return (
            sum([(eval.score - self.mean_score) ** 2 for eval in self.evaluations])
            / len(self.evaluations)
        ) ** 0.5

    @property
    def validation_score_std(self) -> Optional[float]:
        if self.validations is None or len(self.validations) == 0:
            return None

        if self.mean_validation_score is None:
            return None

        return (
            sum(
                [
                    (eval.score - self.mean_validation_score) ** 2
                    for eval in self.validations
                ]
            )
            / len(self.validations)
        ) ** 0.5

    @property
    def zero_score_cases(self) -> bool:
        if self.evaluations is None:
            return False

        return not all([eval.score != 0 for eval in self.evaluations])

    @property
    def trajectory(self) -> list[tuple]:
        trajectory = [tuple([self.mean_score, [self]])]

        # move through generation by generation, starting with the predecessors of the current prompt
        previous_generation = self.predecessors
        while previous_generation is not None and len(previous_generation) > 0:
            previous_gen_scored_prompts = [
                prompt.mean_score
                for prompt in previous_generation
                if prompt.mean_score is not None
            ]

            # average score of the currently previous generation
            if len(previous_gen_scored_prompts) > 0:
                avg_score_prev_gen = sum(previous_gen_scored_prompts) / len(
                    previous_gen_scored_prompts
                )
            else:
                avg_score_prev_gen = None
            trajectory.append(tuple([avg_score_prev_gen, previous_generation]))  # type: ignore

            # the next "previous" generation is the set of all predecessors of the current generation combined
            new_previous_generation = list()
            for prompt in previous_generation:
                if prompt.predecessors is not None:
                    new_previous_generation.extend(prompt.predecessors)
            previous_generation = new_previous_generation

        # return in such an order that the oldest prompts appear in the beggining
        return trajectory[::-1]

    @property
    def generation(self) -> int:
        return len(self.trajectory) - 1

    def __hash__(self):
        return hash(self.prompt)

    def __eq__(self, other):
        if isinstance(other, Prompt):
            return self.prompt == other.prompt
        return False

    def __repr__(self) -> str:
        prompt_repr = text.cap_length(self.prompt.replace("\n", "\\n"), 100)
        mean_score_repr = (
            f"{self.mean_score:.4f}" if self.mean_score is not None else "None"
        )
        mean_val_score_repr = (
            f"{self.mean_validation_score:.4f}"
            if self.mean_validation_score is not None
            else "None"
        )

        return f"Prompt(prompt='{prompt_repr}' (len={len(self.prompt)}), mean_score={mean_score_repr}, mean_validation_score={mean_val_score_repr}, zero_score_cases={self.zero_score_cases})"

    class Config:
        arbitrary_types_allowed = True


PromptSet = list[Prompt]
