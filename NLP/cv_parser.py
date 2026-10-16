import re
import sys
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from NLP.skill_extractor import extract_skills
from NLP.anonymizer import anonymize_text


# ============================================================
# Safe spaced headings
# ============================================================

SAFE_SPACED_HEADINGS = {
    # English
    "SKILLS": "SKILLS",
    "EXPERIENCE": "EXPERIENCE",
    "EDUCATION": "EDUCATION",
    "PROJECT": "PROJECT",
    "PROJECTS": "PROJECTS",
    "LANGUAGE": "LANGUAGE",
    "LANGUAGES": "LANGUAGES",
    "SUMMARY": "SUMMARY",
    "PROFILE": "PROFILE",
    "CONTACT": "CONTACT",
    "CONTACTS": "CONTACTS",
    "CERTIFICATE": "CERTIFICATE",
    "CERTIFICATES": "CERTIFICATES",
    "CERTIFICATION": "CERTIFICATION",
    "CERTIFICATIONS": "CERTIFICATIONS",
    "COURSE": "COURSE",
    "COURSES": "COURSES",
    "ACHIEVEMENTS": "ACHIEVEMENTS",
    "COMPETITIVEPROGRAMMING": "COMPETITIVE PROGRAMMING",

    # Russian
    "НАВЫКИ": "НАВЫКИ",
    "ОПЫТ": "ОПЫТ",
    "ОБРАЗОВАНИЕ": "ОБРАЗОВАНИЕ",
    "ПРОЕКТ": "ПРОЕКТ",
    "ПРОЕКТЫ": "ПРОЕКТЫ",
    "ЯЗЫК": "ЯЗЫК",
    "ЯЗЫКИ": "ЯЗЫКИ",
    "КОНТАКТЫ": "КОНТАКТЫ",

    # Kazakh
    "ДАҒДЫЛАР": "ДАҒДЫЛАР",
    "ТӘЖІРИБЕ": "ТӘЖІРИБЕ",
    "БІЛІМ": "БІЛІМ",
    "БІЛІМІ": "БІЛІМІ",
    "ЖОБА": "ЖОБА",
    "ЖОБАЛАР": "ЖОБАЛАР",
    "ТІЛ": "ТІЛ",
    "ТІЛДЕР": "ТІЛДЕР",
    "ТЕХНИКАЛЫҚ": "ТЕХНИКАЛЫҚ",
    "КӘСІБИТҮЙІНДЕМЕ": "КӘСІБИ ТҮЙІНДЕМЕ",
}


SAFE_COMBINED_SPACED_HEADINGS = {
    "CONTACTPROFILESUMMARY": "CONTACT PROFILE SUMMARY",
    "EDUCATIONPROJECTEXPERIENCE": "EDUCATION PROJECT EXPERIENCE",
    "БІЛІМІКӘСІБИТҮЙІНДЕМЕ": "БІЛІМІ КӘСІБИ ТҮЙІНДЕМЕ",

    # Degrees (fully uppercase spaced text has no visible word boundaries)
    "BACHELOROFSCIENCE": "BACHELOR OF SCIENCE",
    "MASTEROFSCIENCE": "MASTER OF SCIENCE",
}


# ============================================================
# OCR
# ============================================================

def ocr_page(page):
    """
    OCR fallback for pages without selectable text.
    Supports Kazakh, Russian and English.
    """

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )

    return pytesseract.image_to_string(
        image,
        lang="kaz+rus+eng",
    )


# ============================================================
# PDF extraction
# ============================================================

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF.

    First:    PyMuPDF
    Fallback: OCR when a page has no selectable text.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    pages = []

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(document, start=1):

            try:
                text = page.get_text("text", sort=True)

                if not text.strip():
                    print(f"OCR fallback: page {page_number}")
                    text = ocr_page(page)

                if text:
                    pages.append(text)

            except Exception as error:
                print(f"Warning: could not read page {page_number}: {error}")

    return "\n".join(pages)


# ============================================================
# Spaced heading repair (dictionary based)
# ============================================================

def repair_spaced_heading(line):
    """
    Safely repair known letter-spaced headings.

        S K I L L S
        -> SKILLS

        C E R T I F I C A T E S & L A N G U A G E S
        -> CERTIFICATES & LANGUAGES

        C O N T A C T P R O F I L E S U M M A R Y
        -> CONTACT PROFILE SUMMARY

    Unknown or ambiguous spaced text is preserved unchanged.
    """

    if not isinstance(line, str):
        return line

    stripped = line.strip()

    if not stripped:
        return line

    leading_spaces = len(line) - len(line.lstrip())

    def repair_segment(segment):

        segment = segment.strip()

        if not segment:
            return None

        parts = segment.split()

        if len(parts) < 4:
            return None

        # Every token must be exactly one alphabetic character.
        if not all(len(part) == 1 and part.isalpha() for part in parts):
            return None

        candidate = "".join(parts).upper()

        repaired = SAFE_COMBINED_SPACED_HEADINGS.get(candidate)
        if repaired is not None:
            return repaired

        repaired = SAFE_SPACED_HEADINGS.get(candidate)
        if repaired is not None:
            return repaired

        return None

    # 1. Try whole line
    repaired_whole = repair_segment(stripped)

    if repaired_whole is not None:
        return " " * leading_spaces + repaired_whole

    # 2. Joined headings separated by &, / or |
    pieces = re.split(r"(\s*[&|/]\s*)", stripped)

    if len(pieces) == 1:
        return line

    repaired_pieces = []

    for index, piece in enumerate(pieces):

        # Separator
        if index % 2 == 1:
            repaired_pieces.append(f" {piece.strip()} ")
            continue

        repaired_piece = repair_segment(piece)

        # If one segment is unknown, leave the whole line unchanged.
        if repaired_piece is None:
            return line

        repaired_pieces.append(repaired_piece)

    return " " * leading_spaces + "".join(repaired_pieces)


# ============================================================
# Mixed spaced text repair (real words + spaced chains)
# ============================================================

def repair_mixed_spaced(line):
    """
    'B a c h e l o r of S c i e n c e' -> 'Bachelor of Science'

    Joins only chains of 4+ single letters. Real words (like 'of')
    keep their boundaries. If the line has no regular words at all
    (e.g. 'M U R A T B E K ...'), it is left unchanged, because word
    boundaries cannot be recovered.
    """

    if not isinstance(line, str):
        return line

    stripped = line.strip()
    tokens = stripped.split()

    if len(tokens) < 5:
        return line

    def is_single(token):
        return len(token) == 1 and token.isalpha()

    # No regular words -> boundaries cannot be recovered
    if all(is_single(token) for token in tokens):
        return line

    out = []
    run = []

    def flush():
        if len(run) >= 4:
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for token in tokens:
        if is_single(token):
            run.append(token)
        else:
            flush()
            out.append(token)

    flush()

    leading = len(line) - len(line.lstrip())

    return " " * leading + " ".join(out)


def repair_line(line):
    """
    Dictionary repair first, mixed-text repair second.
    """

    repaired = repair_spaced_heading(line)

    if repaired != line:
        return repaired

    return repair_mixed_spaced(line)


# ============================================================
# Cleaning
# ============================================================

def clean_cv_text(text):
    """
    Clean extracted CV text while preserving
    structure and word boundaries for NER.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Repair spaced text line by line
    text = "\n".join(repair_line(line) for line in text.splitlines())

    # Whitespace cleanup
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Complete parser
# ============================================================

def parse_cv(file_path):
    """
    PDF -> PyMuPDF -> OCR fallback -> cleaning
        -> anonymization -> skill extraction
    """

    file_path = Path(file_path)

    raw_text = extract_text_from_pdf(file_path)
    clean_text = clean_cv_text(raw_text)
    anonymized_text = anonymize_text(clean_text)
    skills = extract_skills(anonymized_text)

    return {
        "filename": file_path.name,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "text": anonymized_text,
        "skills": skills,
        "skill_count": len(skills),
    }


# ============================================================
# Display result
# ============================================================

def print_cv_result(result):
    """
    Print safe summary for testing.
    """

    print("\n" + "=" * 70)
    print("CV PARSING RESULT")
    print("=" * 70)

    print(f"\nFile: {result['filename']}")
    print(f"Raw characters: {len(result['raw_text'])}")
    print(f"Clean characters: {len(result['clean_text'])}")
    print(f"Anonymized characters: {len(result['text'])}")
    print(f"Skills detected: {result['skill_count']}")

    print("\nDetected skills:")

    if result["skills"]:
        for skill in result["skills"]:
            print(f"  - {skill}")
    else:
        print("  None")

    print("\n" + "=" * 70)
    print("TEXT PREVIEW")
    print("=" * 70)

    print(result["text"][:1500])

    if len(result["text"]) > 1500:
        print("\n...[text truncated]...")


# ============================================================
# Cleaning regression test
# ============================================================

def run_cleaning_regression_test():
    """
    Free regression test for spaced-text repair.
    """

    cases = {
        "S K I L L S": "SKILLS",
        "C E R T I F I C A T E S": "CERTIFICATES",
        "L A N G U A G E S": "LANGUAGES",
        "C O M P E T I T I V E P R O G R A M M I N G": "COMPETITIVE PROGRAMMING",
        "Ж О Б А Л А Р": "ЖОБАЛАР",
        "Т Е Х Н И К А Л Ы Қ": "ТЕХНИКАЛЫҚ",
        "Д А Ғ Д Ы Л А Р": "ДАҒДЫЛАР",
        "C O N T A C T P R O F I L E S U M M A R Y": "CONTACT PROFILE SUMMARY",
        "E D U C A T I O N P R O J E C T E X P E R I E N C E": "EDUCATION PROJECT EXPERIENCE",
        "Б І Л І М І К Ә С І Б И Т Ү Й І Н Д Е М Е": "БІЛІМІ КӘСІБИ ТҮЙІНДЕМЕ",
        "C E R T I F I C A T E S & L A N G U A G E S": "CERTIFICATES & LANGUAGES",

        # Mixed text: real words keep boundaries
        "B a c h e l o r of S c i e n c e": "Bachelor of Science",

        # Uppercase degree phrase, known from dictionary
        "B A C H E L O R O F S C I E N C E": "BACHELOR OF SCIENCE",

        # Must stay unchanged
        "Python, Java, C#": "Python, Java, C#",
        "M U R A T B E K K A R A K O Z": "M U R A T B E K K A R A K O Z",
    }

    print("=" * 70)
    print("CV CLEANING REGRESSION CHECK")
    print("=" * 70)

    all_passed = True

    for source, expected in cases.items():

        actual = repair_line(source)
        passed = actual == expected

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"

        print(f"{status:<5} | {source!r} -> {actual!r}")

    print("=" * 70)

    if all_passed:
        print("ALL CLEANING CHECKS PASSED")
    else:
        print("SOME CLEANING CHECKS FAILED")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) == 2 and sys.argv[1] == "--test-cleaning":
        run_cleaning_regression_test()
        sys.exit(0)

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python -m NLP.cv_parser path/to/cv.pdf\n\n"
            "Cleaning regression test:\n"
            "python -m NLP.cv_parser --test-cleaning"
        )
        sys.exit(1)

    try:
        result = parse_cv(sys.argv[1])
        print_cv_result(result)

    except Exception as error:
        print(f"\nError while processing CV:\n{error}")
        sys.exit(1)