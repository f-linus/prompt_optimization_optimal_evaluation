from typing import Any


def cap_length(text: Any, max_length=200) -> str:
    """
    Shorten the text to a maximum length.
    """

    text = str(text)

    if len(text) <= max_length:
        return text

    insertion_length = len(" [...] ")
    prefix_ratio = 0.7
    prefix_length = int(max_length * prefix_ratio)
    suffix_length = max_length - prefix_length - insertion_length

    return text[:prefix_length] + " [...] " + text[-suffix_length:]


def remove_line_breaks(text: Any) -> str:
    """
    Remove line breaks from the text.
    """

    text = str(text)
    return text.replace("\n", "\\n")
