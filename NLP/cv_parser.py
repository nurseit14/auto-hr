import re
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from NLP.skill_extractor import extract_skills
from NLP.anonymizer import anonymize_text


def ocr_page(page):
    """
    OCR fallback for pages without selectable text.
    Supports Kazakh, Russian and English.
    """

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    text = pytesseract.image_to_string(
        image,
        lang="kaz+rus+eng"
    )

    return text


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF CV.

    First try normal PyMuPDF text extraction.
    If a page contains no selectable text,
    use OCR fallback.
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

    pages = []

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(
            document,
            start=1
        ):
            try:

                text = page.get_text(
                    "text",
                    sort=True
                )

                # OCR fallback only if normal extraction is empty
                if not text.strip():

                    print(
                        f"OCR fallback: page {page_number}"
                    )

                    text = ocr_page(
                        page
                    )

                if text:
                    pages.append(text)

            except Exception as error:

                print(
                    f"Warning: could not read page "
                    f"{page_number}: {error}"
                )

    return "\n".join(pages)


def repair_spaced_heading(line):
    """
    Repair headings where letters are separated by spaces.

    Examples:
        S K I L L S
        -> SKILLS

        Б І Л І М І
        -> БІЛІМІ

    Normal text such as:
        Python, Java, C#
    is left unchanged.
    """

    if not isinstance(line, str):
        return line

    stripped = line.strip()

    if not stripped:
        return line

    parts = stripped.split()

    if len(parts) < 4:
        return line

    single_character_parts = [
        part
        for part in parts
        if len(part) == 1 and part.isalpha()
    ]

    ratio = (
        len(single_character_parts)
        / len(parts)
    )

    # Only repair lines that strongly look
    # like artificially letter-spaced headings.
    if ratio >= 0.75:

        repaired = "".join(parts)

        leading_spaces = (
            len(line)
            - len(line.lstrip())
        )

        return (
            " " * leading_spaces
            + repaired
        )

    return line


def clean_cv_text(text):
    """
    Clean extracted CV text while preserving
    useful structure for NER.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace(
        "\x00",
        " "
    )

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # --------------------------------------------------
    # Fix known letter-spaced section headings
    # --------------------------------------------------

    heading_replacements = {
        "S K I L L S": "SKILLS",
        "Ж О Б А Л А Р": "ЖОБАЛАР",
        "Т Е Х Н И К А Л Ы Қ": "ТЕХНИКАЛЫҚ",
        "Д А Ғ Д Ы Л А Р": "ДАҒДЫЛАР",
    }

    for old, new in heading_replacements.items():
        text = text.replace(
            old,
            new
        )

    # --------------------------------------------------
    # Repair other strongly spaced headings
    # --------------------------------------------------

    lines = text.splitlines()

    repaired_lines = [
        repair_spaced_heading(line)
        for line in lines
    ]

    text = "\n".join(
        repaired_lines
    )

    # --------------------------------------------------
    # Basic whitespace cleanup
    # --------------------------------------------------

    # Remove trailing spaces before line breaks
    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    # Normalize repeated spaces/tabs inside lines
    text = re.sub(
        r"[ \t]{2,}",
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
    Complete CV parsing pipeline.

    PDF
      -> PyMuPDF extraction
      -> OCR fallback if necessary
      -> cleaned text
      -> anonymized text
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

        "raw_text": raw_text,

        "clean_text": clean_text,

        "text": anonymized_text,

        "skills": skills,

        "skill_count": len(skills),
    }


def print_cv_result(result):
    """
    Print a safe summary for testing.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CV PARSING RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFile: "
        f"{result['filename']}"
    )

    print(
        f"Raw characters: "
        f"{len(result['raw_text'])}"
    )

    print(
        f"Clean characters: "
        f"{len(result['clean_text'])}"
    )

    print(
        f"Anonymized characters: "
        f"{len(result['text'])}"
    )

    print(
        f"Skills detected: "
        f"{result['skill_count']}"
    )

    print(
        "\nDetected skills:"
    )

    if result["skills"]:

        for skill in result["skills"]:

            print(
                f"  - {skill}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEXT PREVIEW"
    )

    print(
        "=" * 70
    )

    preview = (
        result["text"][:1500]
    )

    print(
        preview
    )

    if len(result["text"]) > 1500:

        print(
            "\n...[text truncated]..."
        )


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