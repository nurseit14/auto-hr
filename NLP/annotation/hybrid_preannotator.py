import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from NLP.config import RESUMES_FILE
from NLP.resume_loader import load_resumes

from NLP.annotation.llm_annotator import LLMAnnotator

from NLP.annotation.preannotate_resumes import (
    generate_annotations,
)


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent



OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "annotations"
    / "resume_hybrid_preannotated_tasks.json"
)


# Hybrid V1 strategy:
# HARD_SKILL + SOFT_SKILL -> Rules V2
# EXPERIENCE + EDUCATION + LANG -> LLM V1

RULE_LABELS = {
    "HARD_SKILL",
    "SOFT_SKILL",
}

LLM_LABELS = {
    "EXPERIENCE",
    "EDUCATION",
    "LANG",
}


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Hybrid V1 Label Studio "
            "pre-annotations."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=6,
        help=(
            "First resume number to process. "
            "Default: 6, because resumes 1-5 "
            "are the pilot development set."
        ),
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help=(
            "Last resume number to process. "
            "Example: --end 10"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional custom output JSON path."
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def resume_number(
    resume_id
):
    """
    Convert:
        resume_0006 -> 6
    """

    try:

        return int(
            resume_id.split("_")[-1]
        )

    except Exception:

        return None


def create_label_studio_region(
    start,
    end,
    text,
    label
):
    """
    Convert one exact entity span into
    Label Studio format.
    """

    start = int(start)
    end = int(end)

    return {
        "id": str(uuid.uuid4())[:8],
        "from_name": "label",
        "to_name": "text",
        "type": "labels",

        "value": {
            "start": start,
            "end": end,

            "text": text[
                start:end
            ],

            "labels": [
                label
            ],
        },
    }


# ============================================================
# Rules V2
# ============================================================

def get_rules_regions(
    text
):
    """
    Keep only HARD_SKILL and SOFT_SKILL
    from Rules V2.
    """

    raw_regions = generate_annotations(
        text
    )

    regions = []

    for region in raw_regions:

        value = region.get(
            "value",
            {}
        )

        labels = value.get(
            "labels",
            []
        )

        if not labels:
            continue

        label = labels[0]

        if label not in RULE_LABELS:
            continue

        start = value.get(
            "start"
        )

        end = value.get(
            "end"
        )

        if (
            start is None
            or end is None
        ):
            continue

        try:

            start = int(start)
            end = int(end)

        except (
            TypeError,
            ValueError,
        ):

            continue

        # Basic range validation.
        if (
            start < 0
            or end <= start
            or end > len(text)
        ):
            continue

        actual_text = text[
            start:end
        ]

        regions.append(
            {
                "start": start,
                "end": end,
                "text": actual_text,
                "label": label,
                "source": "rules_v2",
            }
        )

    return regions


# ============================================================
# LLM V1
# ============================================================

def get_llm_regions(
    annotator,
    text
):
    """
    Keep only EXPERIENCE, EDUCATION and LANG
    from LLM V1.

    LLMAnnotator now returns exact global
    character offsets for the specific occurrence
    in context.
    """

    raw_regions = annotator.annotate(
        text,
        verbose=False
    )

    regions = []

    for region in raw_regions:

        label = region.get(
            "label"
        )

        if label not in LLM_LABELS:
            continue

        try:

            start = int(
                region["start"]
            )

            end = int(
                region["end"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

        # Validate range.
        if (
            start < 0
            or end <= start
            or end > len(text)
        ):

            continue

        actual_text = text[
            start:end
        ]

        predicted_text = str(
            region.get(
                "text",
                ""
            )
        )

        # Exact grounding check:
        # offsets must refer to the exact
        # annotated text.
        if (
            actual_text
            != predicted_text
        ):

            continue

        regions.append(
            {
                "start": start,
                "end": end,
                "text": actual_text,
                "label": label,
                "source": "llm_v1",
            }
        )

    return regions


# ============================================================
# Merge
# ============================================================

def merge_regions(
    rules_regions,
    llm_regions
):
    """
    Merge Hybrid V1 predictions.

    Exact duplicates:
        same start + end + label

    are removed.

    Different-label overlaps are preserved
    for human review.
    """

    merged = []

    seen = set()

    for region in (
        rules_regions
        + llm_regions
    ):

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

        merged.append(
            region
        )

    merged.sort(
        key=lambda item: (
            item["start"],
            item["end"],
            item["label"],
        )
    )

    return merged


# ============================================================
# Convert to Label Studio
# ============================================================

def to_label_studio_regions(
    text,
    regions
):

    results = []

    for region in regions:

        start = region[
            "start"
        ]

        end = region[
            "end"
        ]

        label = region[
            "label"
        ]

        # Final validation before JSON export.
        if (
            start < 0
            or end <= start
            or end > len(text)
        ):

            continue

        if (
            text[start:end]
            != region["text"]
        ):

            continue

        results.append(
            create_label_studio_region(
                start=start,
                end=end,
                text=text,
                label=label,
            )
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    print(
        "=" * 70
    )

    print(
        "AUTO-HR HYBRID V1 PRE-ANNOTATOR"
    )

    print(
        "=" * 70
    )

    print(
        "\nHybrid strategy:"
    )

    print(
        "  HARD_SKILL  -> Rules V2"
    )

    print(
        "  SOFT_SKILL  -> Rules V2"
    )

    print(
        "  EXPERIENCE  -> LLM V1"
    )

    print(
        "  EDUCATION   -> LLM V1"
    )

    print(
        "  LANG        -> LLM V1"
    )

    print(
        f"\nStarting from resume: "
        f"{args.start}"
    )

    if args.end is not None:

        print(
            f"Ending at resume: "
            f"{args.end}"
        )

    # --------------------------------------------------------
    # Load resumes
    # --------------------------------------------------------

    if not RESUMES_FILE.exists():

        raise FileNotFoundError(
            f"Resume dataset not found:\n"
            f"{RESUMES_FILE}"
        )

    print(
        f"\nLoading resumes from:\n"
        f"{RESUMES_FILE}"
    )

    df = load_resumes(RESUMES_FILE)

    print(
        f"\nLoaded resumes: "
        f"{len(df)}"
    )

    # Ignore unreadable CVs.
    df = df[
        df["text_length"] > 0
    ].copy()

    selected_rows = []

    for _, row in df.iterrows():

        resume_id = str(
            row[
                "resume_id"
            ]
        )

        number = resume_number(
            resume_id
        )

        if number is None:
            continue

        if number < args.start:
            continue

        if (
            args.end is not None
            and number > args.end
        ):

            continue

        selected_rows.append(
            row
        )

    print(
        f"\nCVs selected: "
        f"{len(selected_rows)}"
    )

    if not selected_rows:

        print(
            "\nNothing to process."
        )

        return

    # --------------------------------------------------------
    # Initialize LLM once
    # --------------------------------------------------------

    try:

        annotator = (
            LLMAnnotator()
        )

    except Exception as error:

        print(
            "\nCould not initialize "
            "LLM annotator:"
        )

        print(
            error
        )

        return

    print(
        f"LLM model: "
        f"{annotator.model}"
    )

    tasks = []

    total_rules = 0
    total_llm = 0
    total_hybrid = 0

    failed = []

    # --------------------------------------------------------
    # Process CVs
    # --------------------------------------------------------

    for index, row in enumerate(
        selected_rows,
        start=1
    ):

        resume_id = str(
            row[
                "resume_id"
            ]
        )

        filename = row.get(
            "filename",
            ""
        )

        if pd.isna(
            filename
        ):

            filename = ""

        filename = str(
            filename
        )

        text = row.get(
            "text",
            ""
        )

        if pd.isna(
            text
        ):

            text = ""

        text = str(
            text
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"[{index}/{len(selected_rows)}] "
            f"{resume_id}"
        )

        print(
            f"File: "
            f"{filename}"
        )

        print(
            f"Characters: "
            f"{len(text)}"
        )

        # ----------------------------------------------------
        # Rules
        # ----------------------------------------------------

        try:

            rules_regions = (
                get_rules_regions(
                    text
                )
            )

        except Exception as error:

            print(
                "Rules V2 failed: "
                f"{error}"
            )

            failed.append(
                resume_id
            )

            continue

        print(
            f"Rules V2 regions: "
            f"{len(rules_regions)}"
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        try:

            llm_regions = (
                get_llm_regions(
                    annotator,
                    text
                )
            )

        except Exception as error:

            print(
                "LLM annotation failed: "
                f"{error}"
            )

            failed.append(
                resume_id
            )

            # IMPORTANT:
            # do not create incomplete Hybrid
            # result when LLM annotation is incomplete.
            continue

        print(
            f"LLM selected regions: "
            f"{len(llm_regions)}"
        )

        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        hybrid_regions = (
            merge_regions(
                rules_regions,
                llm_regions
            )
        )

        print(
            f"Hybrid regions: "
            f"{len(hybrid_regions)}"
        )

        total_rules += len(
            rules_regions
        )

        total_llm += len(
            llm_regions
        )

        total_hybrid += len(
            hybrid_regions
        )

        label_studio_regions = (
            to_label_studio_regions(
                text,
                hybrid_regions
            )
        )

        # Final integrity check.
        if (
            len(label_studio_regions)
            != len(hybrid_regions)
        ):

            print(
                "Hybrid integrity validation failed. "
                "Task was not saved."
            )

            failed.append(
                resume_id
            )

            continue

        # ----------------------------------------------------
        # Label Studio task
        # ----------------------------------------------------

        task = {
            "data": {
                "text":
                    text,

                "resume_id":
                    resume_id,

                "filename":
                    filename,

                "document_id":
                    str(row["document_id"]),

                "source_sha256":
                    str(row["source_sha256"]),

                "text_sha256":
                    str(row["text_sha256"]),

                "text_version":
                    str(row["text_version"]),
            },

            "predictions": [
                {
                    "model_version":
                        "autohr-hybrid-v1-context-offsets",

                    # Not a calibrated probability.
                    "score":
                        0.5,

                    "result":
                        label_studio_regions,
                }
            ],
        }

        tasks.append(
            task
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if args.output:

        output_file = Path(
            args.output
        )

        if not output_file.is_absolute():

            output_file = (
                ROOT_DIR
                / output_file
            )

    else:

        output_file = (
            OUTPUT_FILE
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tasks,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID PRE-ANNOTATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTasks created: "
        f"{len(tasks)}"
    )

    print(
        f"Rules regions used: "
        f"{total_rules}"
    )

    print(
        f"LLM regions used: "
        f"{total_llm}"
    )

    print(
        f"Final hybrid regions: "
        f"{total_hybrid}"
    )

    if tasks:

        print(
            f"Average hybrid regions/CV: "
            f"{total_hybrid / len(tasks):.2f}"
        )

    if failed:

        print(
            "\nFAILED / INCOMPLETE CVs:"
        )

        for resume_id in failed:

            print(
                f"  - {resume_id}"
            )

    print(
        f"\nSaved to:\n"
        f"{output_file}"
    )

    if failed:

        print(
            "\nWARNING:"
        )

        print(
            "Some CVs were not included because "
            "their annotation was incomplete."
        )

        print(
            "Do not treat this output as a complete "
            "batch until failed CVs are reviewed."
        )

    else:

        print(
            "\nAll selected CVs were processed "
            "without incomplete LLM annotations."
        )


if __name__ == "__main__":
    main()