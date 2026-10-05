import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from NLP.annotation.llm_annotator import (
    LLMAnnotator,
    find_entity_occurrences,
)

from NLP.annotation.preannotate_resumes import (
    generate_annotations,
)


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

RESUMES_FILE = (
    ROOT_DIR
    / "data"
    / "resumes"
    / "processed"
    / "resumes.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "annotations"
    / "resume_hybrid_preannotated_tasks.json"
)


# Hybrid V1 strategy determined from pilot development data.

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
    resume_0006 -> 6
    """

    try:

        return int(
            resume_id.split(
                "_"
            )[-1]
        )

    except Exception:

        return None


def create_label_studio_region(
    start,
    end,
    text,
    label
):

    return {
        "id":
            str(uuid.uuid4())[:8],

        "from_name":
            "label",

        "to_name":
            "text",

        "type":
            "labels",

        "value": {
            "start":
                int(start),

            "end":
                int(end),

            "text":
                text[
                    int(start):
                    int(end)
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

    raw_regions = (
        generate_annotations(
            text
        )
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

        regions.append(
            {
                "start":
                    int(start),

                "end":
                    int(end),

                "text":
                    text[
                        int(start):
                        int(end)
                    ],

                "label":
                    label,

                "source":
                    "rules_v2",
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
    """

    entities = annotator.annotate(
        text,
        verbose=False
    )

    # Convert entity strings to exact offsets.
    raw_regions = (
        find_entity_occurrences(
            text,
            entities
        )
    )

    regions = []

    for region in raw_regions:

        label = region.get(
            "label"
        )

        if label not in LLM_LABELS:
            continue

        regions.append(
            {
                "start":
                    int(
                        region["start"]
                    ),

                "end":
                    int(
                        region["end"]
                    ),

                "text":
                    region["text"],

                "label":
                    label,

                "source":
                    "llm_v1",
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

    Duplicate (start, end, label) regions are removed.

    We intentionally do not aggressively resolve
    different-label overlaps here because a human
    reviewer will verify all predictions.
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

        results.append(
            create_label_studio_region(
                start=region[
                    "start"
                ],

                end=region[
                    "end"
                ],

                text=text,

                label=region[
                    "label"
                ],
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

    df = pd.read_csv(
        RESUMES_FILE
    )

    # Ignore unreadable CVs.
    if "character_count" in df.columns:

        df = df[
            df["character_count"] > 0
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

    annotator = (
        LLMAnnotator()
    )

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

        filename = str(
            row.get(
                "filename",
                ""
            )
        )

        text = row.get(
            "text",
            ""
        )

        if pd.isna(text):
            text = ""

        text = str(text)

        print(
            "\n" + "-" * 70
        )

        print(
            f"[{index}/{len(selected_rows)}] "
            f"{resume_id}"
        )

        print(
            f"File: {filename}"
        )

        print(
            f"Characters: "
            f"{len(text)}"
        )

        # ----------------------------------------------------
        # Rules
        # ----------------------------------------------------

        rules_regions = (
            get_rules_regions(
                text
            )
        )

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
                f"LLM annotation failed: "
                f"{error}"
            )

            failed.append(
                resume_id
            )

            # Do NOT silently create a supposedly
            # complete hybrid task without its LLM
            # component.
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
            },

            "predictions": [
                {
                    "model_version":
                        "autohr-hybrid-v1",

                    # This is not calibrated probability.
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

        output_file = OUTPUT_FILE

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
            "\nFAILED CVs:"
        )

        for resume_id in failed:

            print(
                f"  - {resume_id}"
            )

    print(
        f"\nSaved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
