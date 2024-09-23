from prompt_optimization.protocols.prompt_set_modifier import (
    PromptSetModifier,
)


class EvaluatorConfig:
    validation: bool


class Evaluator(PromptSetModifier):
    config: EvaluatorConfig
