import re


# ============================================================
# Patterns
# ============================================================

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

URL_PATTERN = re.compile(
    r"https?://\S+",
    flags=re.IGNORECASE,
)

WWW_PATTERN = re.compile(
    r"\bwww\.\S+",
    flags=re.IGNORECASE,
)

PROFILE_URL_PATTERN = re.compile(
    r"\b(?:github\.com|linkedin\.com)/\S+",
    flags=re.IGNORECASE,
)

TELEGRAM_PATTERN = re.compile(
    r"(?<!\w)@[A-Za-z0-9_]{4,}"
)


# ============================================================
# Phone pattern
# ============================================================

# IMPORTANT:
# Do NOT use \s here because \s also matches newlines.
# We only allow spaces/tabs inside one phone number.
PHONE_CANDIDATE_PATTERN = re.compile(
    r"(?<!\w)"
    r"\+?"
    r"\d"
    r"[\d \t()\-]{5,}"
    r"\d"
    r"(?!\w)"
)


# ============================================================
# Date protection
# ============================================================

DATE_RANGE_PATTERNS = [
    # 2019 - 2023
    re.compile(
        r"^\s*"
        r"(?:19|20)\d{2}"
        r"\s*[-–—]\s*"
        r"(?:"
        r"(?:19|20)\d{2}"
        r"|present"
        r"|current"
        r"|now"
        r"|настоящее\s+время"
        r"|по\s+настоящее\s+время"
        r")"
        r"\s*$",
        flags=re.IGNORECASE,
    ),

    # 09.2022 - 06.2025
    re.compile(
        r"^\s*"
        r"(?:0?[1-9]|1[0-2])"
        r"[./-]"
        r"(?:19|20)\d{2}"
        r"\s*[-–—]\s*"
        r"(?:"
        r"(?:0?[1-9]|1[0-2])"
        r"[./-]"
        r"(?:19|20)\d{2}"
        r"|present"
        r"|current"
        r"|now"
        r"|настоящее\s+время"
        r"|по\s+настоящее\s+время"
        r")"
        r"\s*$",
        flags=re.IGNORECASE,
    ),

    # 2022.09 - 2025.06
    re.compile(
        r"^\s*"
        r"(?:19|20)\d{2}"
        r"[./-]"
        r"(?:0?[1-9]|1[0-2])"
        r"\s*[-–—]\s*"
        r"(?:"
        r"(?:19|20)\d{2}"
        r"[./-]"
        r"(?:0?[1-9]|1[0-2])"
        r"|present"
        r"|current"
        r"|now"
        r"|настоящее\s+время"
        r"|по\s+настоящее\s+время"
        r")"
        r"\s*$",
        flags=re.IGNORECASE,
    ),
]


def is_date_range(value):
    """
    Check whether a phone-like candidate is actually
    an employment or education date range.
    """

    value = value.strip()

    for pattern in DATE_RANGE_PATTERNS:

        if pattern.fullmatch(value):
            return True

    return False


# ============================================================
# Phone validation
# ============================================================

def is_phone_candidate(value):
    """
    Decide whether a candidate is probably a real phone.

    Rules:
    - date ranges must stay unchanged;
    - phone must contain 10-15 digits.
    """

    value = value.strip()

    # Never replace employment/education date ranges.
    if is_date_range(value):
        return False

    digits = re.sub(
        r"\D",
        "",
        value,
    )

    digit_count = len(
        digits
    )

    # Phone numbers normally have 10-15 digits.
    if digit_count < 10:
        return False

    if digit_count > 15:
        return False

    return True


def mask_phone_match(match):
    """
    Replace only validated telephone numbers.
    """

    value = match.group(0)

    if is_phone_candidate(
        value
    ):
        return "[PHONE]"

    return value


# ============================================================
# Letter-spaced candidate name protection
# ============================================================

# Spaced lines that are section headings, NOT names.
# Stored WITHOUT spaces, uppercase, to compare with joined letters.
# NOTE: do not import these from cv_parser (cv_parser imports
# anonymizer -> circular import). Keep this list in sync manually.
KNOWN_SPACED_HEADINGS = {
    # English
    "SKILLS", "EXPERIENCE", "EDUCATION", "PROJECT", "PROJECTS",
    "LANGUAGE", "LANGUAGES", "SUMMARY", "PROFILE", "CONTACT",
    "CONTACTS", "CERTIFICATE", "CERTIFICATES", "CERTIFICATION",
    "CERTIFICATIONS", "COURSE", "COURSES", "ACHIEVEMENTS",
    "REFERENCES", "COMPETITIVEPROGRAMMING", "CONTACTPROFILESUMMARY",
    "EDUCATIONPROJECTEXPERIENCE", "BACHELOROFSCIENCE",
    "MASTEROFSCIENCE",

    # Russian
    "НАВЫКИ", "ОПЫТ", "ОБРАЗОВАНИЕ", "ПРОЕКТ", "ПРОЕКТЫ",
    "ЯЗЫК", "ЯЗЫКИ", "КОНТАКТЫ",

    # Kazakh
    "ДАҒДЫЛАР", "ТӘЖІРИБЕ", "БІЛІМ", "БІЛІМІ", "ЖОБА", "ЖОБАЛАР",
    "ТІЛ", "ТІЛДЕР", "ТЕХНИКАЛЫҚ", "КӘСІБИТҮЙІНДЕМЕ",
    "БІЛІМІКӘСІБИТҮЙІНДЕМЕ",
}

# A name is only expected in the CV header.
SPACED_NAME_MAX_LINES = 5

# Minimum number of single letters to treat a line as a name.
SPACED_NAME_MIN_LETTERS = 8

# Share of tokens that must be single letters.
SPACED_NAME_MIN_RATIO = 0.9


def _is_single_letter(token):

    return len(token) == 1 and token.isalpha()


def mask_spaced_name(text):
    """
    Mask a letter-spaced candidate name in the CV header.

        M U R A T B E K K A R A K O Z
        -> [NAME]

    Rules:
    - only the first few non-empty lines are checked;
    - the line must be (almost) only single letters;
    - it must have enough letters;
    - it must not be a known section heading.

    If a name is found, its spaced and joined forms are also
    removed from the rest of the text (footer, signature, etc.).
    """

    lines = text.split("\n")

    seen = 0
    found_names = []

    for index, line in enumerate(lines):

        if not line.strip():
            continue

        seen += 1

        if seen > SPACED_NAME_MAX_LINES:
            break

        tokens = line.split()

        singles = [
            token
            for token in tokens
            if _is_single_letter(token)
        ]

        if len(singles) < SPACED_NAME_MIN_LETTERS:
            continue

        if len(singles) / len(tokens) < SPACED_NAME_MIN_RATIO:
            continue

        joined = "".join(singles).upper()

        if joined in KNOWN_SPACED_HEADINGS:
            continue

        found_names.append(joined)

        lines[index] = "[NAME]"

    text = "\n".join(lines)

    # Remove the same name elsewhere, in spaced or joined form.
    for joined in found_names:

        pattern = re.compile(
            r"(?<!\w)"
            + r"[ \t]*".join(
                re.escape(char)
                for char in joined
            )
            + r"(?!\w)",
            flags=re.IGNORECASE,
        )

        text = pattern.sub(
            "[NAME]",
            text,
        )

    return text


# ============================================================
# Main anonymizer
# ============================================================

def anonymize_text(text):
    """
    Mask common personally identifying information.

    Employment and education dates are intentionally
    preserved because they are required for NER.
    """

    if not isinstance(
        text,
        str,
    ):
        return ""

    # --------------------------------------------------------
    # Letter-spaced candidate name
    # --------------------------------------------------------

    text = mask_spaced_name(
        text
    )

    # --------------------------------------------------------
    # Emails
    # --------------------------------------------------------

    text = EMAIL_PATTERN.sub(
        "[EMAIL]",
        text,
    )

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    text = URL_PATTERN.sub(
        "[URL]",
        text,
    )

    text = WWW_PATTERN.sub(
        "[URL]",
        text,
    )

    text = PROFILE_URL_PATTERN.sub(
        "[URL]",
        text,
    )

    # --------------------------------------------------------
    # Phones
    # --------------------------------------------------------

    text = PHONE_CANDIDATE_PATTERN.sub(
        mask_phone_match,
        text,
    )

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    text = TELEGRAM_PATTERN.sub(
        "[CONTACT]",
        text,
    )

    # --------------------------------------------------------
    # Whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# Regression test
# ============================================================

if __name__ == "__main__":

    sample = """
John Example
john@example.com
+7 777 123 45 67
8 701 222 33 44

github.com/john/example
linkedin.com/in/john-example

Python Backend Developer
Python, Docker, PostgreSQL

Employment: 2019 - 2023
Employment: 09.2022 - 06.2025
Education: 2020 - 2024
"""

    spaced_sample = """
M U R A T B E K K A R A K O Z
CYBERSECURITYSTUDENTINTERN MK

C O N T A C T P R O F I L E S U M M A R Y
muratbek@example.com
+7 777 123 45 67

Employment: 2022 - 2025

Signature: MURATBEKKARAKOZ
"""

    heading_sample = """
C O N T A C T P R O F I L E S U M M A R Y
R E F E R E N C E S
Python, Docker
"""

    print("=" * 70)
    print("ORIGINAL")
    print("=" * 70)

    print(sample)

    result = anonymize_text(
        sample
    )

    spaced_result = anonymize_text(
        spaced_sample
    )

    heading_result = anonymize_text(
        heading_sample
    )

    print("\n" + "=" * 70)
    print("ANONYMIZED")
    print("=" * 70)

    print(result)

    print("\n" + "=" * 70)
    print("ANONYMIZED (spaced name sample)")
    print("=" * 70)

    print(spaced_result)

    print("\n" + "=" * 70)
    print("REGRESSION CHECK")
    print("=" * 70)

    checks = {
        "Email masked":
            "[EMAIL]" in result,

        "Phone +7 masked":
            "+7 777 123 45 67" not in result,

        "Phone 8 masked":
            "8 701 222 33 44" not in result,

        "2019 - 2023 preserved":
            "2019 - 2023" in result,

        "09.2022 - 06.2025 preserved":
            "09.2022 - 06.2025" in result,

        "2020 - 2024 preserved":
            "2020 - 2024" in result,

        "Spaced name masked":
            "M U R A T B E K" not in spaced_result
            and "[NAME]" in spaced_result,

        "Joined name masked (signature)":
            "MURATBEKKARAKOZ" not in spaced_result,

        "Spaced sample: phone masked":
            "+7 777 123 45 67" not in spaced_result,

        "Spaced sample: date preserved":
            "2022 - 2025" in spaced_result,

        "Spaced sample: next line kept":
            "CYBERSECURITYSTUDENTINTERN MK" in spaced_result,

        "Spaced heading not treated as name":
            "[NAME]" not in heading_result,

        "Plain text not treated as name":
            "[NAME]" not in result,
    }

    for name, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{status:<5} | {name}"
        )