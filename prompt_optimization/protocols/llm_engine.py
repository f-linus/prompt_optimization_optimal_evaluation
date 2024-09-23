class LLMEngine:
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    history: list[list[dict[str, str]]] = []
    usage: dict[str, int] = {
        "responses_total": 0,
        "responses_cached": 0,
        "completions_total": 0,
        "completions_cached": 0,
    }

    async def non_stream_create(self, messages: list[dict]) -> list[str]: ...

    def _add_to_history(self, messages: list[dict], completions: list[str]):
        self.history.append(messages + [{"role": "assistant", "content": completions}])
