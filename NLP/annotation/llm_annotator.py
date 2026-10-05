import json
import os
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from NLP.annotation.llm_annotation_prompt import SYSTEM_PROMPT


# ============================================================
# Configuration
# ============================================================

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

RESUMES_FILE = (
    ROOT_DIR
    / "data"
    / "resumes"
    / "processed"
    / "resumes.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "annotations"
    / "llm_predictions"
)

DEFAULT_MODEL = os.getenv(
    "OPENAI_ANNOTATION_MODEL",
    os.getenv(
        "OPENAI_ASSISTANT_MODEL",
        "gpt-6-luna"
    )
)

VALID_LABELS = {
    "HARD_SKILL",
    "SOFT_SKILL",
    "EXPERIENCE",
    "EDUCATION",
    "LANG",
}


# Approximate maximum size of each CV chunk.
CHUNK_SIZE = 1000

# Give the model enough output room for one small chunk,
# without allowing unnecessarily huge responses.
MAX_OUTPUT_TOKENS_PER_CHUNK = 1500


# ============================================================
# Structured Output Schema
# ============================================================

class NEREntity(BaseModel):
    text: str

    label: Literal[
        "HARD_SKILL",
        "SOFT_SKILL",
        "EXPERIENCE",
        "EDUCATION",
        "LANG",
    ]


class NERAnnotationResult(BaseModel):
    entities: list[NEREntity]


# ============================================================
# Text Chunking
# ============================================================

def split_text_into_chunks(
    text,
    max_chars=CHUNK_SIZE
):
    """
    Split CV text into reasonably sized chunks.

    Prefer splitting on newline boundaries so that sections
    and entity phrases are less likely to be cut in half.
    """

    text = (text or "").strip()

    if not text:
        return []

    lines = text.splitlines()

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:

        # +1 represents the newline we will restore.
        line_length = len(line) + 1

        # If one individual line is extremely long,
        # handle it separately.
        if line_length > max_chars:

            # First save any existing chunk.
            if current_lines:

                chunks.append(
                    "\n".join(current_lines)
                )

                current_lines = []
                current_length = 0

            # Split the long line into smaller pieces.
            start = 0

            while start < len(line):

                end = min(
                    start + max_chars,
                    len(line)
                )

                chunks.append(
                    line[start:end]
                )

                start = end

            continue

        # Would this line make the current chunk too large?
        if (
            current_lines
            and current_length + line_length > max_chars
        ):

            chunks.append(
                "\n".join(current_lines)
            )

            current_lines = []
            current_length = 0

        current_lines.append(
            line
        )

        current_length += line_length

    if current_lines:

        chunks.append(
            "\n".join(current_lines)
        )

    return [
        chunk
        for chunk in chunks
        if chunk.strip()
    ]


# ============================================================
# LLM Annotator
# ============================================================

class LLMAnnotator:

    def __init__(self):

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "OPENAI_API_KEY was not found. "
                "Add it to your .env file."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = DEFAULT_MODEL

    # --------------------------------------------------------
    # Validate one chunk's predictions
    # --------------------------------------------------------

    @staticmethod
    def _validate_entities(
        entities,
        source_text
    ):

        valid = []

        seen = set()

        for entity in entities:

            if not isinstance(
                entity,
                dict
            ):
                continue

            entity_text = str(
                entity.get(
                    "text",
                    ""
                )
            ).strip()

            label = str(
                entity.get(
                    "label",
                    ""
                )
            ).strip()

            if not entity_text:
                continue

            if label not in VALID_LABELS:
                continue

            # Grounding check:
            # the prediction must literally occur
            # inside the chunk supplied to the model.
            if entity_text not in source_text:
                continue

            key = (
                entity_text,
                label
            )

            # The LLM does NOT need to return Python
            # five times. Python will find all occurrences.
            if key in seen:
                continue

            seen.add(
                key
            )

            valid.append(
                {
                    "text": entity_text,
                    "label": label,
                }
            )

        return valid

    # --------------------------------------------------------
    # Annotate ONE chunk
    # --------------------------------------------------------

    def annotate_chunk(
        self,
        chunk_text
    ):

        user_input = f"""
Annotate the following PART of an anonymized CV.

Identify entities belonging to:

HARD_SKILL
SOFT_SKILL
EXPERIENCE
EDUCATION
LANG

Follow the supplied annotation guidelines exactly.

IMPORTANT OUTPUT RULES:

- Return each UNIQUE entity text/label combination only once
  for this chunk.
- If "Python" appears multiple times, return Python only once.
- Python code will later locate every occurrence.
- Return entity text EXACTLY as written.
- Do not normalize spelling.
- Do not translate.
- Do not paraphrase.
- Do not invent entities.
- Do not annotate section headings.
- The text below may be only one section of a larger CV.

CV CHUNK
========

{chunk_text}
""".strip()

        response = (
            self.client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=user_input,
                text_format=NERAnnotationResult,
                max_output_tokens=(
                    MAX_OUTPUT_TOKENS_PER_CHUNK
                ),
            )
        )

        parsed = response.output_parsed

        if parsed is None:

            raise ValueError(
                "The model did not return "
                "structured annotations."
            )

        entities = [
            {
                "text": entity.text,
                "label": entity.label,
            }
            for entity in parsed.entities
        ]

        return self._validate_entities(
            entities,
            chunk_text
        )

    # --------------------------------------------------------
    # Annotate COMPLETE CV
    # --------------------------------------------------------

    def annotate(
        self,
        cv_text,
        verbose=True
    ):

        cv_text = (
            cv_text
            or ""
        ).strip()

        if not cv_text:
            return []

        chunks = split_text_into_chunks(
            cv_text
        )

        if verbose:

            print(
                "\nSplitting CV into chunks..."
            )

            print(
                f"Chunks: {len(chunks)}"
            )

        all_entities = []

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            if verbose:

                print(
                    f"\n[{index}/{len(chunks)}] "
                    f"Annotating chunk "
                    f"({len(chunk)} characters)..."
                )

            try:

                entities = (
                    self.annotate_chunk(
                        chunk
                    )
                )

            except Exception as error:

                print(
                    f"  Chunk {index} failed: "
                    f"{error}"
                )

                continue

            all_entities.extend(
                entities
            )

            if verbose:

                print(
                    f"  Entities: "
                    f"{len(entities)}"
                )

        # ----------------------------------------------------
        # Deduplicate predictions across chunks
        # ----------------------------------------------------

        unique_entities = []

        seen = set()

        for entity in all_entities:

            key = (
                entity["text"],
                entity["label"]
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique_entities.append(
                entity
            )

        return unique_entities


# ============================================================
# Convert Entity Text -> Character Offsets
# ============================================================

def find_entity_occurrences(
    cv_text,
    entities
):
    """
    Convert predicted entity texts into character spans.

    Improvements over V1:
    - finds repeated occurrences;
    - avoids matching short entities inside larger words;
    - resolves overlapping entities;
    - prefers the longest meaningful span when two
      entities with the same label overlap.

    Examples:

        PostgreSQL -> HARD_SKILL

    will not additionally produce:

        SQL -> HARD_SKILL

    from inside the word PostgreSQL.

    Similarly:

        GitHub Actions

    is preferred over nested:

        GitHub
        Git
    """

    import re

    candidates = []

    # ==================================================
    # 1. Find candidate occurrences
    # ==================================================

    for entity in entities:

        entity_text = entity[
            "text"
        ].strip()

        label = entity[
            "label"
        ]

        if not entity_text:
            continue

        # Escape special regex characters such as:
        # C++, C#, JPA/Hibernate, etc.
        escaped = re.escape(
            entity_text
        )

        # Prevent matching inside another alphanumeric word.
        #
        # Example:
        #
        # SQL
        #
        # should match:
        #
        # SQL queries
        #
        # but NOT:
        #
        # PostgreSQL
        #
        pattern = (
            rf"(?<![\w])"
            rf"{escaped}"
            rf"(?![\w])"
        )

        for match in re.finditer(
            pattern,
            cv_text,
            flags=re.IGNORECASE
        ):

            start = match.start()
            end = match.end()

            candidates.append(
                {
                    "start":
                        start,

                    "end":
                        end,

                    # Always preserve the exact text
                    # from the original CV.
                    "text":
                        cv_text[start:end],

                    "label":
                        label,
                }
            )

    # ==================================================
    # 2. Remove exact duplicate predictions
    # ==================================================

    unique_candidates = []

    seen = set()

    for candidate in candidates:

        key = (
            candidate["start"],
            candidate["end"],
            candidate["label"],
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_candidates.append(
            candidate
        )

    # ==================================================
    # 3. Prefer longer spans
    # ==================================================
    #
    # Example:
    #
    # Git
    # GitHub
    # GitHub Actions
    #
    # At the same location, GitHub Actions should
    # be considered first.

    unique_candidates.sort(
        key=lambda item: (
            -(item["end"] - item["start"]),
            item["start"],
        )
    )

    selected = []

    # ==================================================
    # 4. Resolve overlaps
    # ==================================================

    for candidate in unique_candidates:

        candidate_start = candidate[
            "start"
        ]

        candidate_end = candidate[
            "end"
        ]

        candidate_label = candidate[
            "label"
        ]

        should_keep = True

        for existing in selected:

            existing_start = existing[
                "start"
            ]

            existing_end = existing[
                "end"
            ]

            existing_label = existing[
                "label"
            ]

            # Do the spans overlap?
            overlap = (
                candidate_start
                < existing_end
                and
                candidate_end
                > existing_start
            )

            if not overlap:
                continue

            # --------------------------------------------------
            # Same label + overlapping span
            # --------------------------------------------------
            #
            # Prefer the longer span.
            #
            # Since candidates are already sorted by length,
            # the existing span is normally longer.
            #
            # Example:
            #
            # HARD_SKILL GitHub Actions
            #
            # wins over:
            #
            # HARD_SKILL GitHub
            # HARD_SKILL Git

            if (
                candidate_label
                == existing_label
            ):

                should_keep = False
                break

            # --------------------------------------------------
            # Different labels
            # --------------------------------------------------
            #
            # For now we keep different-label overlaps.
            #
            # Example:
            #
            # if future annotation rules legitimately allow
            # different entity categories to overlap, we do
            # not silently destroy that information here.

        if should_keep:

            selected.append(
                candidate
            )

    # ==================================================
    # 5. Sort back into document order
    # ==================================================

    selected.sort(
        key=lambda item: (
            item["start"],
            item["end"],
            item["label"],
        )
    )

    return selected


# ============================================================
# Load Resume
# ============================================================

def load_resume(
    resume_id
):

    if not RESUMES_FILE.exists():

        raise FileNotFoundError(
            f"Resume dataset not found:\n"
            f"{RESUMES_FILE}"
        )

    df = pd.read_csv(
        RESUMES_FILE
    )

    if "resume_id" not in df.columns:

        raise ValueError(
            "resumes.csv does not contain "
            "a resume_id column."
        )

    matches = df[
        df["resume_id"]
        == resume_id
    ]

    if matches.empty:

        raise ValueError(
            f"Resume not found: "
            f"{resume_id}"
        )

    row = matches.iloc[0]

    text = row.get(
        "text",
        ""
    )

    if pd.isna(text):
        text = ""

    filename = row.get(
        "filename",
        ""
    )

    if pd.isna(filename):
        filename = ""

    return {
        "resume_id": resume_id,
        "filename": str(filename),
        "text": str(text),
    }


# ============================================================
# Save Prediction
# ============================================================

def save_prediction(
    resume_id,
    model,
    regions
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"{resume_id}.json"
    )

    result = {
        "resume_id": resume_id,
        "model": model,
        "entity_count": len(regions),
        "entities": regions,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_file


# ============================================================
# Display Results
# ============================================================

def print_predictions(
    regions
):

    print(
        "\nPredictions:"
    )

    print(
        "-" * 70
    )

    if not regions:

        print(
            "No entities detected."
        )

        return

    for region in regions:

        print(
            f"{region['label']:<12} "
            f"{region['start']:>5}:"
            f"{region['end']:<5} "
            f"{region['text']}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "AUTO-HR LLM ANNOTATOR"
    )

    print(
        "=" * 70
    )

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python -m "
            "NLP.annotation.llm_annotator "
            "resume_0001"
        )

        return

    resume_id = sys.argv[
        1
    ]

    # --------------------------------------------------------
    # Load resume
    # --------------------------------------------------------

    try:

        resume = load_resume(
            resume_id
        )

    except Exception as error:

        print(
            f"\nCould not load resume:\n"
            f"{error}"
        )

        return

    print(
        f"\nResume: "
        f"{resume_id}"
    )

    print(
        f"File: "
        f"{resume['filename']}"
    )

    print(
        f"Characters: "
        f"{len(resume['text'])}"
    )

    # --------------------------------------------------------
    # Initialize LLM
    # --------------------------------------------------------

    try:

        annotator = (
            LLMAnnotator()
        )

    except Exception as error:

        print(
            f"\nCould not initialize "
            f"LLM annotator:\n"
            f"{error}"
        )

        return

    print(
        f"Model: "
        f"{annotator.model}"
    )

    print(
        "\nGenerating LLM annotations..."
    )

    # --------------------------------------------------------
    # Annotate
    # --------------------------------------------------------

    try:

        entities = (
            annotator.annotate(
                resume["text"],
                verbose=True
            )
        )

    except Exception as error:

        print(
            "\nLLM annotation failed:"
        )

        print(
            error
        )

        return

    print(
        "\nMerging predictions..."
    )

    # --------------------------------------------------------
    # Locate every occurrence in original CV
    # --------------------------------------------------------

    regions = (
        find_entity_occurrences(
            resume["text"],
            entities
        )
    )

    print(
        f"\nUnique entity predictions: "
        f"{len(entities)}"
    )

    print(
        f"Located annotation regions: "
        f"{len(regions)}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_predictions(
        regions
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        save_prediction(
            resume_id,
            annotator.model,
            regions
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANNOTATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()