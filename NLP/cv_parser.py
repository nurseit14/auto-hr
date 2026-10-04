import re
from pathlib import Path

from pypdf import PdfReader

from NLP.skill_extractor import extract_skills
from NLP.anonymizer import anonymize_text

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF CV.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"CV file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        try:
            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception as error:
            print(
                f"Warning: could not read page "
                f"{page_number}: {error}"
            )

    return "\n".join(pages)


def clean_cv_text(text):
    """
    Basic cleaning while preserving useful CV content.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize spaces/tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Avoid excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def parse_cv(file_path):
    """
    Complete V1 CV parsing pipeline.

    PDF
      -> text
      -> cleaned text
      -> extracted skills
    """

    file_path = Path(file_path)

    raw_text = extract_text_from_pdf(
        file_path
    )

    clean_text = clean_cv_text(
        raw_text
    )

    anonymized_text = anonymize_text(
        clean_text
    )

    skills = extract_skills(
        anonymized_text
    )

    return {
        "filename": file_path.name,
        "text": anonymized_text,
        "skills": skills,
        "skill_count": len(skills),
    }


def print_cv_result(result):
    """
    Print a safe summary for testing.
    """

    print("\n" + "=" * 70)
    print("CV PARSING RESULT")
    print("=" * 70)

    print(
        f"\nFile: {result['filename']}"
    )

    print(
        f"Characters extracted: "
        f"{len(result['text'])}"
    )

    print(
        f"Skills detected: "
        f"{result['skill_count']}"
    )

    print("\nDetected skills:")

    if result["skills"]:
        for skill in result["skills"]:
            print(f"  - {skill}")
    else:
        print("  None")

    print("\n" + "=" * 70)
    print("TEXT PREVIEW")
    print("=" * 70)

    # Only show first 1500 characters for debugging
    preview = result["text"][:1500]

    print(preview)

    if len(result["text"]) > 1500:
        print("\n...[text truncated]...")


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python -m NLP.cv_parser "
            "path/to/cv.pdf"
        )

        sys.exit(1)

    cv_path = sys.argv[1]

    try:

        result = parse_cv(
            cv_path
        )

        print_cv_result(
            result
        )

    except Exception as error:

        print(
            f"\nError while processing CV:\n"
            f"{error}"
        )

        sys.exit(1)
