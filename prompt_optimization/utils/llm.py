import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


def _get_token_idx_from_position(logprobs, position):
    idx = 0
    current_length = 0
    while current_length + len(logprobs[idx].token) <= position:
        current_length += len(logprobs[idx].token)
        idx += 1
    return idx


def _get_expected_numerical_token(logprobs_token, temperature=1.0):
    top_logprobs = logprobs_token.top_logprobs

    # remove all tokens that are not numbers
    def is_valid_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    top_logprobs = [t for t in top_logprobs if is_valid_number(t.token)]

    values = np.array([float(t.token) for t in top_logprobs])

    log_probs = np.array([t.logprob for t in top_logprobs]) / temperature
    probs = np.exp(log_probs - log_probs.max())
    probs = probs / probs.sum()

    expectation = np.sum(values * probs)

    logger.info(
        f"Tokens {values} with probs {probs} mapped to expectation {expectation:.3f} (temp. used: {temperature})"
    )
    return expectation


def logprobs_chatcompletion_to_expected_number_completion(
    completion_obj, temperature=1.0
) -> list:
    pattern = r"\b\d+\.\d+\b"

    new_completions = []
    for completion in completion_obj.choices:
        msg = completion.message.content
        matches = re.finditer(pattern, msg)

        # replace every integer with its expectation
        in_between_text = []
        expectations = []
        last_match_end = 0
        for match in matches:
            in_between_text.append(msg[last_match_end : match.start()])
            last_match_end = match.end()

            idx = _get_token_idx_from_position(
                completion.logprobs.content, match.start()
            )
            token = completion.logprobs.content[idx]
            expectations.append(
                _get_expected_numerical_token(token, temperature=temperature)
            )

        in_between_text.append(msg[last_match_end:])

        expectations = [str(e) for e in expectations]
        expectations.append("")

        new_message = ""
        for text, expectation in zip(in_between_text, expectations):
            new_message += text + str(expectation)

        new_completions.append(new_message)
    return new_completions
