import json
import os
import re
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

# IMPORTANT:
# All NLP scripts use the same current dataset.
RESUMES_FILE = (
    ROOT_DIR
    / "data"
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
        "gpt-6-luna",
    ),
)

VALID_LABELS = {
    "HARD_SKILL",
    "SOFT_SKILL",
    "EXPERIENCE",
    "EDUCATION",
    "LANG",
}

# Small chunks reduce context ambiguity and API cost.
CHUNK_SIZE = 1000

# Enough room for structured NER output without
# unnecessarily large generations.
MAX_OUTPUT_TOKENS_PER_CHUNK = 800


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

    # Character offsets RELATIVE TO THE CURRENT CHUNK.
    start: int
    end: int


class NERAnnotationResult(BaseModel):
    entities: list[NEREntity]


# ============================================================
# Text Chunking
# ============================================================

def split_text_into_chunks(
    text,
    max_chars=CHUNK_SIZE,
):
    """
    Split CV text into reasonably sized chunks.

    Prefer newline boundaries so sections and entity phrases
    are less likely to be split.

    Each returned item contains:
        text
        start
        end

    start/end refer to the ORIGINAL CV text.
    """

    text = text or ""

    if not text.strip():
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        # Skip leading newlines between chunks.
        while (
            start < text_length
            and text[start] == "\n"
        ):
            start += 1

        if start >= text_length:
            break

        proposed_end = min(
            start + max_chars,
            text_length,
        )

        if proposed_end < text_length:

            # Prefer splitting at the last newline.
            newline_pos = text.rfind(
                "\n",
                start,
                proposed_end,
            )

            if newline_pos > start:
                end = newline_pos

            else:
                # If no newline exists, prefer whitespace.
                whitespace_pos = text.rfind(
                    " ",
                    start,
                    proposed_end,
                )

                if whitespace_pos > start:
                    end = whitespace_pos

                else:
                    end = proposed_end

        else:
            end = text_length

        chunk_text = text[start:end]

        if chunk_text.strip():

            chunks.append(
                {
                    "text": chunk_text,
                    "start": start,
                    "end": end,
                }
            )

        if end <= start:
            # Safety guard.
            start += 1
        else:
            start = end

    return chunks


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
    # Validate one chunk
    # --------------------------------------------------------

    @staticmethod
    def _validate_entities(
        entities,
        source_text,
    ):
        """
        Validate that the LLM returned real exact spans
        inside the chunk.

        This prevents global text search from incorrectly
        assigning one entity to unrelated repetitions.
        """

        valid = []
        seen = set()

        for entity in entities:

            if not isinstance(
                entity,
                dict,
            ):
                continue

            entity_text = str(
                entity.get(
                    "text",
                    "",
                )
            )

            label = str(
                entity.get(
                    "label",
                    "",
                )
            ).strip()

            try:
                start = int(
                    entity.get(
                        "start"
                    )
                )

                end = int(
                    entity.get(
                        "end"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if not entity_text:
                continue

            if label not in VALID_LABELS:
                continue

            if (
                start < 0
                or end <= start
                or end > len(source_text)
            ):
                continue

            # Critical offset validation.
            actual_text = source_text[
                start:end
            ]

            if actual_text != entity_text:
                continue

            key = (
                start,
                end,
                label,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            valid.append(
                {
                    "text": actual_text,
                    "label": label,
                    "start": start,
                    "end": end,
                }
            )

        return valid

    # --------------------------------------------------------
    # Annotate one chunk
    # --------------------------------------------------------

    def annotate_chunk(
        self,
        chunk_text,
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

IMPORTANT OFFSET RULES:

- For every entity return:
  text
  label
  start
  end

- start and end are character offsets relative ONLY
  to the CV CHUNK below.

- Python slicing must satisfy exactly:

  chunk_text[start:end] == text

- Return the exact occurrence that is valid in context.

- Do NOT return an entity simply because the same word
  appears elsewhere.

IMPORTANT ANNOTATION RULES:

- Return each actual entity occurrence separately.
- Return entity text EXACTLY as written.
- Do not normalize spelling.
- Do not translate.
- Do not paraphrase.
- Do not invent entities.
- Do not annotate section headings.
- Desired position is NOT EXPERIENCE.
- Company name alone is NOT EXPERIENCE.
- Date alone is NOT EXPERIENCE.
- University name alone is NOT EDUCATION.
- Language mentioned only as course language is NOT
  automatically a candidate LANG entity.
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
                "start": entity.start,
                "end": entity.end,
            }
            for entity in parsed.entities
        ]

        return self._validate_entities(
            entities,
            chunk_text,
        )

    # --------------------------------------------------------
    # Annotate complete CV
    # --------------------------------------------------------

    def annotate(
        self,
        cv_text,
        verbose=True,
    ):
        """
        Annotate a complete CV.

        Returns exact GLOBAL character offsets.

        IMPORTANT:
        If ANY chunk fails, the whole CV annotation fails.
        This prevents partial LLM results from silently
        becoming ordinary Hybrid predictions.
        """

        cv_text = (
            cv_text
            or ""
        )

        if not cv_text.strip():
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

        all_regions = []
        failed_chunks = []

        for index, chunk_info in enumerate(
            chunks,
            start=1,
        ):

            chunk_text = chunk_info[
                "text"
            ]

            chunk_start = chunk_info[
                "start"
            ]

            if verbose:

                print(
                    f"\n[{index}/{len(chunks)}] "
                    f"Annotating chunk "
                    f"({len(chunk_text)} characters)..."
                )

            try:

                local_entities = (
                    self.annotate_chunk(
                        chunk_text
                    )
                )

            except Exception as error:

                failed_chunks.append(
                    {
                        "chunk": index,
                        "error": str(error),
                    }
                )

                print(
                    f"  Chunk {index} failed: "
                    f"{error}"
                )

                continue

            for entity in local_entities:

                global_start = (
                    chunk_start
                    + entity["start"]
                )

                global_end = (
                    chunk_start
                    + entity["end"]
                )

                actual_text = cv_text[
                    global_start:
                    global_end
                ]

                # Final grounding check against full CV.
                if actual_text != entity["text"]:
                    continue

                all_regions.append(
                    {
                        "start":
                            global_start,

                        "end":
                            global_end,

                        "text":
                            actual_text,

                        "label":
                            entity["label"],
                    }
                )

            if verbose:

                print(
                    f"  Entities: "
                    f"{len(local_entities)}"
                )

        # ----------------------------------------------------
        # B09: never silently accept partial LLM output
        # ----------------------------------------------------

        if failed_chunks:

            failed_numbers = [
                str(
                    item["chunk"]
                )
                for item in failed_chunks
            ]

            raise RuntimeError(
                "LLM annotation is incomplete. "
                "Failed chunks: "
                + ", ".join(
                    failed_numbers
                )
            )

        # ----------------------------------------------------
        # Remove exact duplicates only
        # ----------------------------------------------------

        unique_regions = []
        seen = set()

        for region in all_regions:

            key = (
                region["start"],
                region["end"],
                region["label"],
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique_regions.append(
                region
            )

        unique_regions.sort(
            key=lambda item: (
                item["start"],
                item["end"],
                item["label"],
            )
        )

        return unique_regions


# ============================================================
# Compatibility Helper
# ============================================================

def find_entity_occurrences(
    cv_text,
    entities,
):
    """
    Convert entities to character regions.

    New LLMAnnotator output already contains exact offsets,
    so those offsets are preserved.

    A fallback exists for old text+label predictions,
    but new annotation should use exact offsets.
    """

    regions = []

    # --------------------------------------------------------
    # New format: exact offsets already available
    # --------------------------------------------------------

    for entity in entities:

        if (
            "start" in entity
            and "end" in entity
        ):

            try:
                start = int(
                    entity["start"]
                )

                end = int(
                    entity["end"]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                start < 0
                or end <= start
                or end > len(cv_text)
            ):
                continue

            actual_text = cv_text[
                start:end
            ]

            entity_text = str(
                entity.get(
                    "text",
                    ""
                )
            )

            if actual_text != entity_text:
                continue

            label = str(
                entity.get(
                    "label",
                    ""
                )
            )

            if label not in VALID_LABELS:
                continue

            regions.append(
                {
                    "start": start,
                    "end": end,
                    "text": actual_text,
                    "label": label,
                }
            )

    if regions:

        regions.sort(
            key=lambda item: (
                item["start"],
                item["end"],
                item["label"],
            )
        )

        return regions

    # --------------------------------------------------------
    # Legacy fallback
    # --------------------------------------------------------
    #
    # Kept only so older saved predictions can still be read.
    # Do NOT use this path for new human-reviewed batches.
    # --------------------------------------------------------

    candidates = []

    for entity in entities:

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

        if (
            not entity_text
            or label not in VALID_LABELS
        ):
            continue

        escaped = re.escape(
            entity_text
        )

        pattern = (
            rf"(?<![\w])"
            rf"{escaped}"
            rf"(?![\w])"
        )

        for match in re.finditer(
            pattern,
            cv_text,
            flags=re.IGNORECASE,
        ):

            start = match.start()
            end = match.end()

            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "text":
                        cv_text[start:end],
                    "label": label,
                }
            )

    unique = []
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

        unique.append(
            candidate
        )

    unique.sort(
        key=lambda item: (
            item["start"],
            item["end"],
            item["label"],
        )
    )

    return unique


# ============================================================
# Load Resume
# ============================================================

def load_resume(
    resume_id,
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
    regions,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"{resume_id}.json"
    )

    result = {
        "resume_id":
            resume_id,

        "model":
            model,

        "status":
            "complete",

        "entity_count":
            len(regions),

        "entities":
            regions,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_file


# ============================================================
# Display Results
# ============================================================

def print_predictions(
    regions,
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

        regions = (
            annotator.annotate(
                resume["text"],
                verbose=True,
            )
        )

    except Exception as error:

        print(
            "\nLLM annotation failed:"
        )

        print(
            error
        )

        print(
            "\nNo prediction file was saved "
            "because the CV annotation "
            "was incomplete."
        )

        return

    print(
        f"\nLocated annotation regions: "
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
            regions,
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