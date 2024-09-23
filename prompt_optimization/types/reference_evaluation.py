from prompt_optimization.types.evaluation import Evaluation
from prompt_optimization.types.reference import Reference


class ReferenceEvaluation(Evaluation):
    reference: Reference
    actual: str
    extracted_reference: dict | float | str
    extracted_actual: dict | float | str

    def __repr__(self) -> str:
        context_snippet = self.context[:150].replace("\n", "\\n") + " ..."
        return f"Evaluation(score={self.score}, extracted_actual={self.extracted_actual}, extracted_reference={self.extracted_reference}, context='{context_snippet}')"
