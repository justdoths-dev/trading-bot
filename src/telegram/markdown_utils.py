"""Shared Markdown utilities for Telegram messages."""

def escape_markdown(value: str) -> str:
    if not value:
        return ""

    escaped = value
    for char in (
        "\\",
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ):
        escaped = escaped.replace(char, f"\\{char}")

    return escaped