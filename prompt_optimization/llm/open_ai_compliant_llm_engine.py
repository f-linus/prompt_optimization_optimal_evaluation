import json
import logging
import os
import random
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from prompt_optimization.caching.disk_cache_client import DiskCacheClient
from prompt_optimization.protocols.llm_engine import LLMEngine

MAX_TOKENS_DEFAULT = 4096
MAX_CONCURRENT_REQUESTS = 200
CACHE_ENABLED = True

cache_client = DiskCacheClient(
    cache_name=os.path.join(os.path.dirname(__file__), "openai_compliant_llm_cache")
)

logger = logging.getLogger(__name__)


class OpenAICompliantLLMEngine(LLMEngine):
    """LLM client for which OpenAI-type endpoints, requests are cached on disk."""

    concurrent_requests = {}
    clients = {}

    def __init__(
        self, base_url: str, api_keys: list[str], model: str, temperature: float = 0.6
    ):
        for api_key in api_keys:
            if api_key not in self.clients:
                self.clients[api_key] = AsyncOpenAI(
                    base_url=base_url, api_key=api_key, max_retries=1
                )
            if api_key not in self.concurrent_requests:
                self.concurrent_requests[api_key] = 0

        self.base_url = self.clients[api_keys[0]].base_url
        self.model = model
        self.temperature = temperature

    async def non_stream_create(
        self, messages: list[dict], max_tries=10, starting_backoff=20
    ) -> list[str]:
        unique_request_repr = {
            "base_url": str(self.base_url),
            "self_model": self.model,
            "self_temperature": self.temperature,
            "messages": messages,
            "max_tokens": MAX_TOKENS_DEFAULT,
        }
        cache_key = json.dumps(unique_request_repr, sort_keys=True)

        cached_response = await cache_client.get(cache_key)
        if cached_response is not None and CACHE_ENABLED:
            logger.info(f"Cache hit from key: {cache_key[:100]}...")
            completions = [choice.message.content for choice in cached_response.choices]  # type: ignore
            self._add_to_history(messages, completions)

            # add counter
            self.usage["responses_total"] += 1
            self.usage["completions_total"] += len(completions)
            self.usage["responses_cached"] += 1
            self.usage["completions_cached"] += len(completions)
            return completions

        logger.info(f"Cache miss for key: {cache_key[:100]}..., fetching from API.")

        # cache response and return
        response = None
        backoff = starting_backoff
        for i in range(max_tries):
            # Find the minimum number of concurrent requests
            min_concurrent = min(self.concurrent_requests.values())

            # Get all keys (API keys) with the minimum number of concurrent requests
            min_req_keys = [
                key
                for key, value in self.concurrent_requests.items()
                if value == min_concurrent
            ]

            # Randomly select one of the keys with minimum concurrent requests
            min_req_key = random.choice(min_req_keys)

            client = self.clients[min_req_key]

            # if we are at max concurrent requests, wait for a bit
            while self.concurrent_requests[min_req_key] >= MAX_CONCURRENT_REQUESTS:
                logger.info(
                    f"Max concurrent requests reached for {min_req_key}, waiting 10s..."
                )
                time.sleep(10)

            try:
                self.concurrent_requests[min_req_key] += 1
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=MAX_TOKENS_DEFAULT,
                )
                self.concurrent_requests[min_req_key] -= 1
                break
            except (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            ) as e:
                logger.error(
                    f"LLM call error: {e} Will try again in {backoff} seconds (attempt {i+1}/{max_tries})"
                )
                logger.info(self.concurrent_requests)
                self.concurrent_requests[min_req_key] -= 1
                time.sleep(backoff)
                backoff *= 2

        if response is None:
            raise Exception(f"Failed to fetch from API after {max_tries} attempts")

        # add to history
        self._add_to_history(messages, response)

        # add usage
        self.input_tokens_used += response.usage.prompt_tokens
        self.output_tokens_used += response.usage.completion_tokens

        await cache_client.set(cache_key, response)

        # add counter
        self.usage["responses_total"] += 1
        self.usage["completions_total"] += len(response.choices)
        return [choice.message.content for choice in response.choices]
