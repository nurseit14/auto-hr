import re


def anonymize_text(text):
    """
    Remove/mask common personally identifying information
    from extracted resume text.

    This is a baseline anonymizer and should still be
    followed by manual checking for research data.
    """

    if not isinstance(text, str):
        return ""

    # --------------------------------------------------
    # Email addresses
    # --------------------------------------------------

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text
    )

    # --------------------------------------------------
    # URLs
    # --------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        "[URL]",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bwww\.\S+",
        "[URL]",
        text,
        flags=re.IGNORECASE
    )

    # Common profile URLs written without https://
    text = re.sub(
        r"\b(?:github\.com|linkedin\.com)/\S+",
        "[URL]",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------
    # Phone numbers
    # --------------------------------------------------

    phone_pattern = (
        r"(?<!\w)"
        r"(?:\+?\d[\d\s().-]{7,}\d)"
        r"(?!\w)"
    )

    text = re.sub(
        phone_pattern,
        "[PHONE]",
        text
    )

    # --------------------------------------------------
    # Telegram handles
    # --------------------------------------------------

    text = re.sub(
        r"(?<!\w)@[A-Za-z0-9_]{4,}",
        "[CONTACT]",
        text
    )

    # --------------------------------------------------
    # Normalize spaces
    # --------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


if __name__ == "__main__":

    sample = """
    John Example
    john@example.com
    +7 777 123 45 67
    github.com/john/example
    linkedin.com/in/john-example

    Python Backend Developer
    Python, Docker, PostgreSQL
    """

    print("ORIGINAL:")
    print(sample)

    print("\nANONYMIZED:")
    print(anonymize_text(sample))
