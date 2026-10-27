import math
import re
import sys
from pathlib import Path
from statistics import median

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

    # Headings that used to appear glued to a neighbouring column.
    # With column-aware extraction they arrive as separate lines.
    "PROFILESUMMARY": "PROFILE SUMMARY",
    "PROJECTEXPERIENCE": "PROJECT EXPERIENCE",

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


# Exact-match phrases whose word boundaries cannot be recovered
# from spaced letters. Only add phrases you have seen in real CVs.
SAFE_COMBINED_SPACED_HEADINGS = {
    "CONTACTPROFILESUMMARY": "CONTACT PROFILE SUMMARY",
    "EDUCATIONPROJECTEXPERIENCE": "EDUCATION PROJECT EXPERIENCE",
    "БІЛІМІКӘСІБИТҮЙІНДЕМЕ": "БІЛІМІ КӘСІБИ ТҮЙІНДЕМЕ",

    # Degrees
    "BACHELOROFSCIENCE": "BACHELOR OF SCIENCE",
    "MASTEROFSCIENCE": "MASTER OF SCIENCE",

    # Job titles seen in real CVs
    "CYBERSECURITYSTUDENTINTERN": "CYBERSECURITY STUDENT INTERN",
}


# ============================================================
# Column detection settings (block based)
# ============================================================

# Pages with fewer text blocks are never treated as multi-column.
COLUMN_MIN_BLOCKS = 6

# Search for the gutter only inside this part of the page width.
COLUMN_GUTTER_ZONE = (0.15, 0.85)

# Minimum gutter width: max(10 pt, 2% of page width).
COLUMN_GUTTER_MIN_WIDTH_RATIO = 0.02
COLUMN_GUTTER_MIN_WIDTH_PT = 10

# A few blocks (full-width headers) may cross the gutter.
COLUMN_GUTTER_CROSSING_RATIO = 0.12

# Each side must hold at least this share of the blocks.
COLUMN_MIN_SIDE_BLOCK_RATIO = 0.10

# Each side must span at least this share of the text height.
COLUMN_MIN_SIDE_HEIGHT_RATIO = 0.25

# Each side must be at least this wide (share of page width).
# Protects "date | description" timeline layouts from being split.
COLUMN_MIN_SIDE_WIDTH_RATIO = 0.20

# If full-width bands cover more than this share of the text
# height, the page is not really two-column.
COLUMN_MAX_BAND_RATIO = 0.5

# A right-column block above the left column is read first
# (e.g. a name above a sidebar) only if it is separated from
# the rest of its column by at least this many line heights.
COLUMN_LEAD_GAP_LINES = 1.5


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
# Block geometry helpers
# ============================================================
# PyMuPDF block tuple:
#   (x0, y0, x1, y1, text, block_no, block_type)
# block_type 0 = text, 1 = image

def _center_x(block):
    return (block[0] + block[2]) / 2


def _center_y(block):
    return (block[1] + block[3]) / 2


def _vertical_extent(blocks):
    if not blocks:
        return 0.0
    return max(b[3] for b in blocks) - min(b[1] for b in blocks)


def _horizontal_extent(blocks):
    if not blocks:
        return 0.0
    return max(b[2] for b in blocks) - min(b[0] for b in blocks)


def _line_count(block):
    lines = [line for line in block[4].split("\n") if line.strip()]
    return max(1, len(lines))


def _typical_line_height(blocks):
    heights = [
        (b[3] - b[1]) / _line_count(b)
        for b in blocks
        if b[3] > b[1]
    ]

    return median(heights) if heights else 10.0


def blocks_to_text(blocks, typical):
    """
    Join blocks (top to bottom, then left to right) into text.
    A blank line is inserted where the vertical gap between
    two blocks looks like a paragraph break.
    """

    ordered = sorted(blocks, key=lambda b: (b[1], b[0]))

    out = []
    previous_bottom = None

    for block in ordered:

        text = block[4].strip("\n")

        if not text.strip():
            continue

        if previous_bottom is not None:
            if block[1] - previous_bottom > 0.8 * typical:
                out.append("")

        out.append(text.rstrip())
        previous_bottom = block[3]

    return "\n".join(out)


# ============================================================
# Column detection
# ============================================================

def find_gutter(blocks, page_width):
    """
    Find a vertical gutter between two text columns.

    Returns (x_start, x_end) or None for single-column pages.
    """

    count = len(blocks)

    if count < COLUMN_MIN_BLOCKS or page_width <= 0:
        return None

    size = int(math.ceil(page_width)) + 2
    coverage = [0] * size

    for block in blocks:

        start = max(0, int(math.floor(block[0])))
        end = min(size - 1, int(math.ceil(block[2])))

        for i in range(start, end):
            coverage[i] += 1

    allowed = max(1, int(COLUMN_GUTTER_CROSSING_RATIO * count))

    low = int(COLUMN_GUTTER_ZONE[0] * page_width)
    high = int(COLUMN_GUTTER_ZONE[1] * page_width)

    runs = []
    run_start = None

    for i in range(low, high):

        if coverage[i] <= allowed:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i))
                run_start = None

    if run_start is not None:
        runs.append((run_start, high))

    min_width = max(
        COLUMN_GUTTER_MIN_WIDTH_PT,
        COLUMN_GUTTER_MIN_WIDTH_RATIO * page_width,
    )

    total_height = _vertical_extent(blocks)

    # Widest candidate first.
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)

    for start, end in runs:

        if end - start < min_width:
            continue

        middle = (start + end) / 2

        left = [b for b in blocks if _center_x(b) < middle]
        right = [b for b in blocks if _center_x(b) >= middle]

        if len(left) < COLUMN_MIN_SIDE_BLOCK_RATIO * count:
            continue

        if len(right) < COLUMN_MIN_SIDE_BLOCK_RATIO * count:
            continue

        if _vertical_extent(left) < COLUMN_MIN_SIDE_HEIGHT_RATIO * total_height:
            continue

        if _vertical_extent(right) < COLUMN_MIN_SIDE_HEIGHT_RATIO * total_height:
            continue

        min_side_width = COLUMN_MIN_SIDE_WIDTH_RATIO * page_width

        if _horizontal_extent(left) < min_side_width:
            continue

        if _horizontal_extent(right) < min_side_width:
            continue

        return (start, end)

    return None


def _take_right_lead(left, right, typical):
    """
    Right-column blocks that sit clearly above the whole left
    column (for example a name above a sidebar) must be read
    first, otherwise they would end up after the entire left
    column.

    Returns (lead_blocks, remaining_right_blocks).
    """

    if not left or not right:
        return [], right

    top_left = min(b[1] for b in left)
    top_right = min(b[1] for b in right)

    if top_right >= top_left - 1:
        return [], right

    lead = [b for b in right if b[3] <= top_left + 1]
    rest = [b for b in right if b[3] > top_left + 1]

    if not lead:
        return [], right

    # A heading directly above its own content is not a header:
    # require a visible gap before the rest of the right column.
    if rest:

        gap = min(b[1] for b in rest) - max(b[3] for b in lead)

        if gap < COLUMN_LEAD_GAP_LINES * typical:
            return [], right

    return lead, rest


def classify_two_column_blocks(blocks, gutter):
    """
    Split the blocks of a two-column page into reading groups.

    Returns (header, lead, left, right), or None when the page
    does not look two-column after all.

    header : full-width blocks that have no column content above
             them (name, title), plus anything on the same rows
    lead   : right-column blocks clearly above the left column
    left   : the whole left column; a full-width block lower on
             the page stays here, in vertical order
    right  : the whole right column

    Reading order is: header, lead, left, right.
    """

    gutter_start, gutter_end = gutter
    middle = (gutter_start + gutter_end) / 2

    typical = _typical_line_height(blocks)

    # Full-width = spans the WHOLE gutter. A block that only pokes
    # into the (tolerant) gutter belongs to its own column.
    crossing = [
        b for b in blocks
        if b[0] < gutter_start and b[2] > gutter_end
    ]

    crossing_ids = {id(b) for b in crossing}

    columns = [b for b in blocks if id(b) not in crossing_ids]

    if not columns:
        return None

    # Header = full-width blocks with no column content above them.
    header_crossing = [
        c for c in crossing
        if all(_center_y(b) >= c[1] - 1 for b in columns)
    ]

    bands = []

    for block in sorted(header_crossing, key=lambda b: b[1]):

        if bands and block[1] <= bands[-1][1] + 0.5:
            bands[-1][1] = max(bands[-1][1], block[3])
        else:
            bands.append([block[1], block[3]])

    total_height = _vertical_extent(blocks)
    band_height = sum(bottom - top for top, bottom in bands)

    if total_height > 0 and band_height > COLUMN_MAX_BAND_RATIO * total_height:
        return None

    header = []
    left = []
    right = []

    first_header_bottom = (
        max(bottom for _, bottom in bands)
        if bands else None
    )

    for block in blocks:

        center = _center_y(block)

        in_header_row = any(
            top - 1 <= center <= bottom + 1
            for top, bottom in bands
        )

        above_header = (
            first_header_bottom is not None
            and block[3] <= first_header_bottom + 1
        )

        if in_header_row or above_header:
            header.append(block)
        elif id(block) in crossing_ids:
            left.append(block)
        elif _center_x(block) < middle:
            left.append(block)
        else:
            right.append(block)

    # Keep oversized title elements together with a common
    # full-width header, even when their bounding boxes are tall.
    if header:
        header_top = min(b[1] for b in header)
        header_bottom = max(b[3] for b in header)

        moved = []

        for group in (left, right):
            for block in group[:]:
                overlaps_header = (
                    block[1] < header_bottom
                    and block[3] > header_top
                )

                upper_page = block[1] < 0.25 * max(
                    b[3] for b in blocks
                )

                oversized = (
                    block[3] - block[1] >= 3 * typical
                )

                if overlaps_header and upper_page and oversized:
                    group.remove(block)
                    moved.append(block)

        header.extend(moved)

    lead, right = _take_right_lead(left, right, typical)

    return header, lead, left, right


def order_two_column_blocks(blocks, gutter):
    """
    Build text for a two-column page from PyMuPDF blocks:
    header, then the whole left column, then the whole right
    column (never interleaved by vertical position).

    Returns None if the page does not look two-column.
    """

    groups = classify_two_column_blocks(blocks, gutter)

    if groups is None:
        return None

    typical = _typical_line_height(blocks)

    chunks = []

    for part in groups:

        text = blocks_to_text(part, typical)

        if text.strip():
            chunks.append(text)

    return "\n\n".join(chunks)


def extract_page_text(page):
    """
    Extract text from one page.

    Returns (text, layout) where layout is
    "single" or "two_column".

    Single-column pages (and pages where column detection is
    not confident) use the original PyMuPDF text mode, so their
    extraction does not change.
    """

    # Use individual text lines for column detection.
    # PyMuPDF can merge left and right columns into one block.
    blocks = []

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "")
                for span in line.get("spans", [])
            )

            if not text.strip():
                continue

            x0, y0, x1, y1 = line["bbox"]

            blocks.append(
                (x0, y0, x1, y1, text, len(blocks), 0)
            )

    gutter = None

    if blocks:
        gutter = find_gutter(blocks, page.rect.width)

    if gutter is not None:

        text = order_two_column_blocks(blocks, gutter)

        if text is not None and text.strip():
            return text, "two_column"

    return page.get_text("text", sort=True), "single"


# ============================================================
# PDF extraction
# ============================================================

def extract_pdf_with_layout(file_path):
    """
    Extract text from PDF and report the layout of each page.

    Returns (text, layouts), where layouts is a list like:
        [{"page": 1, "layout": "two_column"}, ...]

    Layout values: "single", "two_column", "ocr".
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    pages = []
    layouts = []

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(document, start=1):

            try:
                text, layout = extract_page_text(page)

                if not text.strip():
                    print(f"OCR fallback: page {page_number}")
                    text = ocr_page(page)
                    layout = "ocr"

                if text.strip():
                    pages.append(text)
                    layouts.append(
                        {
                            "page": page_number,
                            "layout": layout,
                        }
                    )
                else:
                    print(
                        f"ERROR: page {page_number} "
                        "has no extractable text after OCR"
                    )
                    layouts.append(
                        {
                            "page": page_number,
                            "layout": "failed",
                            "error": "OCR returned empty text",
                        }
                    )

            except Exception as error:
                print(
                    f"ERROR: could not read page "
                    f"{page_number}: {error}"
                )
                layouts.append(
                    {
                        "page": page_number,
                        "layout": "failed",
                        "error": str(error),
                    }
                )

    return "\n".join(pages), layouts


def extract_text_from_pdf(file_path):
    """
    Backward-compatible wrapper: returns only the text.
    """

    text, _ = extract_pdf_with_layout(file_path)

    return text


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

        P R O F I L E S U M M A R Y
        -> PROFILE SUMMARY

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
    PDF -> PyMuPDF (column-aware) -> OCR fallback -> cleaning
        -> anonymization -> skill extraction

    review_required is True when reading order is heuristic:
    OCR pages and multi-column pages.
    """

    file_path = Path(file_path)

    raw_text, layouts = extract_pdf_with_layout(file_path)
    clean_text = clean_cv_text(raw_text)
    anonymized_text = anonymize_text(clean_text)
    skills = extract_skills(anonymized_text)

    review_required = any(
        item["layout"] in ("ocr", "two_column", "failed")
        for item in layouts
    )

    return {
        "filename": file_path.name,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "text": anonymized_text,
        "skills": skills,
        "skill_count": len(skills),
        "layouts": layouts,
        "review_required": review_required,
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

    layouts = ", ".join(
        f"p{item['page']}={item['layout']}"
        for item in result["layouts"]
    )

    print(f"Layouts: {layouts or 'none'}")
    print(f"Review required: {result['review_required']}")

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

        # Headings that now arrive as separate column lines
        "C O N T A C T": "CONTACT",
        "P R O F I L E S U M M A R Y": "PROFILE SUMMARY",
        "P R O J E C T E X P E R I E N C E": "PROJECT EXPERIENCE",

        # Job title with no recoverable word boundaries
        "C Y B E R S E C U R I T Y S T U D E N T I N T E R N":
            "CYBERSECURITY STUDENT INTERN",

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

    return all_passed


# ============================================================
# Column regression test (synthetic PDF, no files needed)
# ============================================================

def _put(page, x, y, text, size=10):
    page.insert_text((x, y), text, fontsize=size)


def _build_band_page():
    """
    full-width header
    left 01-06  | right 01-06
    full-width line in the middle of the page
    left 07-12  | right 07-12

    Expected: header, whole left column (the middle line stays in
    it, in vertical order), whole right column.
    """

    document = pymupdf.open()
    page = document.new_page(width=595, height=842)

    _put(
        page, 40, 50,
        "HEADER Jane Example Cybersecurity Student Intern Portfolio Header",
        size=12,
    )

    for i in range(6):

        y = 90 + 26 * i

        _put(page, 40, y, f"LEFT column line number {i + 1:02d} sample")
        _put(page, 340, y, f"RIGHT column line number {i + 1:02d} sample")

    _put(
        page, 40, 270,
        "BAND full width line crossing both columns of this page here and some more words",
    )

    for i in range(6, 12):

        y = 310 + 26 * (i - 6)

        _put(page, 40, y, f"LEFT column line number {i + 1:02d} sample")
        _put(page, 340, y, f"RIGHT column line number {i + 1:02d} sample")

    return document, page


def _build_right_header_page():
    """
    The name sits in the RIGHT column above the sidebar:

                         | NAMER ...
                         | TITLER ...
        left 01-08       | right 01-08
    """

    document = pymupdf.open()
    page = document.new_page(width=595, height=842)

    _put(page, 340, 50, "NAMER Jane Example", size=14)
    _put(page, 340, 72, "TITLER Student Intern", size=11)

    for i in range(8):

        y = 140 + 26 * i

        _put(page, 40, y, f"LEFT column line number {i + 1:02d} sample")
        _put(page, 340, y, f"RIGHT column line number {i + 1:02d} sample")

    return document, page


def _build_single_column_page():

    document = pymupdf.open()
    page = document.new_page(width=595, height=842)

    for i in range(40):
        _put(
            page, 50, 60 + 15 * i,
            f"Single column line {i + 1:02d} with enough words to look like "
            f"a normal paragraph of a resume",
            size=9,
        )

    return document, page


def run_column_regression_test():
    """
    Checks reading order on synthetic pages:
    1. full-width header and band between two columns;
    2. name in the right column above the sidebar;
    3. single-column page must not be split.
    """

    print("=" * 70)
    print("CV COLUMN ORDER REGRESSION CHECK")
    print("=" * 70)

    def in_order(text, tokens):
        positions = [text.find(token) for token in tokens]
        return all(p >= 0 for p in positions) and positions == sorted(positions)

    checks = {}
    failures = []

    # --- 1. bands -------------------------------------------------

    document, page = _build_band_page()
    text, layout = extract_page_text(page)
    document.close()

    checks["Bands: layout detected as two_column"] = layout == "two_column"

    checks["Bands: reading order correct"] = in_order(
        text,
        [
            "HEADER Jane Example",
            "LEFT column line number 01",
            "LEFT column line number 06",
            "BAND full width line",
            "LEFT column line number 07",
            "LEFT column line number 12",
            "RIGHT column line number 01",
            "RIGHT column line number 12",
        ],
    )

    checks["Bands: left and right never share a line"] = not any(
        "LEFT" in line and "RIGHT" in line
        for line in text.splitlines()
    )

    if not all(v for k, v in checks.items() if k.startswith("Bands")):
        failures.append(("band page", layout, text))

    # --- 2. right-column header -------------------------------------

    document, page = _build_right_header_page()
    text, layout = extract_page_text(page)
    document.close()

    checks["Right header: layout detected as two_column"] = (
        layout == "two_column"
    )

    checks["Right header: name read before left column"] = in_order(
        text,
        [
            "NAMER Jane Example",
            "LEFT column line number 01",
            "LEFT column line number 08",
            "RIGHT column line number 01",
            "RIGHT column line number 08",
        ],
    )

    if not all(v for k, v in checks.items() if k.startswith("Right header")):
        failures.append(("right header page", layout, text))

    # --- 3. single column --------------------------------------------

    document, page = _build_single_column_page()
    _, layout = extract_page_text(page)
    document.close()

    checks["Single column page stays single"] = layout == "single"

    # --- report ---------------------------------------------------------

    all_passed = True

    for name, passed in checks.items():

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"

        print(f"{status:<5} | {name}")

    print("=" * 70)

    if all_passed:
        print("ALL COLUMN CHECKS PASSED")
    else:
        print("SOME COLUMN CHECKS FAILED")

        for title, layout, text in failures:
            print(f"\n--- {title}: layout={layout} ---\n")
            print(text)

    return all_passed


# ============================================================
# Layout debugging
# ============================================================

def debug_layout(file_path):
    """
    Print every text block of every page with its role in the
    reading order, to see why a CV is read in a wrong order.

        HEADER  full-width block above the columns
        LEAD    right-column block read before the left column
        LEFT    left column
        RIGHT   right column
        -       page is read as single column (no gutter found)
    """

    file_path = Path(file_path)

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(document, start=1):

            width = page.rect.width

            blocks = [
                b for b in page.get_text("blocks")
                if b[6] == 0 and b[4].strip()
            ]

            gutter = find_gutter(blocks, width) if blocks else None

            print("=" * 70)
            print(
                f"PAGE {page_number} | width={width:.0f} | "
                f"blocks={len(blocks)} | "
                f"gutter={'none' if gutter is None else gutter}"
            )
            print("=" * 70)

            roles = {}
            order = []

            if gutter is not None:

                groups = classify_two_column_blocks(blocks, gutter)

                if groups is None:
                    print("(bands cover too much of the page -> single)")
                else:
                    for name, part in zip(
                        ("HEADER", "LEAD", "LEFT", "RIGHT"),
                        groups,
                    ):
                        for block in part:
                            roles[id(block)] = name

                        order.extend(
                            sorted(part, key=lambda b: (b[1], b[0]))
                        )

            if not order:
                order = sorted(blocks, key=lambda b: (b[1], b[0]))

            for number, block in enumerate(order, start=1):

                role = roles.get(id(block), "-")

                preview = " / ".join(
                    line.strip()
                    for line in block[4].split("\n")
                    if line.strip()
                )[:60]

                print(
                    f"{number:>3} {role:<6} "
                    f"x={block[0]:6.1f}-{block[2]:6.1f} "
                    f"y={block[1]:6.1f}-{block[3]:6.1f} | {preview}"
                )

            print()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) == 2 and sys.argv[1] == "--test-cleaning":
        run_cleaning_regression_test()
        sys.exit(0)

    if len(sys.argv) == 2 and sys.argv[1] == "--test-columns":
        run_column_regression_test()
        sys.exit(0)

    if len(sys.argv) == 3 and sys.argv[1] == "--debug-layout":
        debug_layout(sys.argv[2])
        sys.exit(0)

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python -m NLP.cv_parser path/to/cv.pdf\n\n"
            "Cleaning regression test:\n"
            "python -m NLP.cv_parser --test-cleaning\n\n"
            "Column order regression test:\n"
            "python -m NLP.cv_parser --test-columns\n\n"
            "Block layout dump for one CV:\n"
            "python -m NLP.cv_parser --debug-layout path/to/cv.pdf"
        )
        sys.exit(1)

    try:
        result = parse_cv(sys.argv[1])
        print_cv_result(result)

    except Exception as error:
        print(f"\nError while processing CV:\n{error}")
        sys.exit(1)