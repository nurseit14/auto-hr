import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from NLP.annotation.preannotate_resumes import generate_annotations


# ============================================================
# Paths / Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

GROUND_TRUTH_FILE = (
    ROOT_DIR
    / "data"
    / "annotations"
    / "pilot_annotations.json"
)

LLM_DIR = (
    ROOT_DIR
    / "data"
    / "annotations"
    / "llm_predictions"
)

RESUMES_FILE = (
    ROOT_DIR
    / "data"
    / "resumes"
    / "processed"
    / "resumes.csv"
)

LABELS = [
    "HARD_SKILL",
    "SOFT_SKILL",
    "EXPERIENCE",
    "EDUCATION",
    "LANG",
]


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AUTO-HR pre-annotation "
            "against human ground truth."
        )
    )

    parser.add_argument(
        "--method",
        choices=[
            "llm",
            "rules",
            "hybrid",
        ],
        required=True,
        help=(
            "Pre-annotation method to evaluate."
        ),
    )

    return parser.parse_args()


# ============================================================
# Human Ground Truth
# ============================================================

def load_ground_truth():

    if not GROUND_TRUTH_FILE.exists():

        raise FileNotFoundError(
            f"Ground truth not found:\n"
            f"{GROUND_TRUTH_FILE}"
        )

    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        tasks = json.load(file)

    ground_truth = {}

    for task in tasks:

        resume_id = (
            task.get("data", {})
            .get("resume_id")
        )

        if not resume_id:
            continue

        annotations = task.get(
            "annotations",
            []
        )

        if not annotations:
            continue

        # Use latest completed human annotation.
        annotation = annotations[-1]

        entities = set()

        for result in annotation.get(
            "result",
            []
        ):

            value = result.get(
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

            if label not in LABELS:
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

            entities.add(
                (
                    int(start),
                    int(end),
                    label,
                )
            )

        ground_truth[
            resume_id
        ] = entities

    return ground_truth


# ============================================================
# LLM Predictions
# ============================================================

def load_llm_predictions(
    allowed_resume_ids=None
):

    if not LLM_DIR.exists():

        raise FileNotFoundError(
            f"LLM prediction directory "
            f"not found:\n{LLM_DIR}"
        )

    predictions = {}

    for file_path in sorted(
        LLM_DIR.glob(
            "resume_*.json"
        )
    ):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        resume_id = data.get(
            "resume_id"
        )

        if not resume_id:
            continue

        if (
            allowed_resume_ids is not None
            and resume_id
            not in allowed_resume_ids
        ):
            continue

        entities = set()

        for entity in data.get(
            "entities",
            []
        ):

            label = entity.get(
                "label"
            )

            if label not in LABELS:
                continue

            start = entity.get(
                "start"
            )

            end = entity.get(
                "end"
            )

            if (
                start is None
                or end is None
            ):
                continue

            entities.add(
                (
                    int(start),
                    int(end),
                    label,
                )
            )

        predictions[
            resume_id
        ] = entities

    return predictions


# ============================================================
# Rules V2 Predictions
# ============================================================

def load_rules_predictions(
    allowed_resume_ids
):

    if not RESUMES_FILE.exists():

        raise FileNotFoundError(
            f"Resume dataset not found:\n"
            f"{RESUMES_FILE}"
        )

    df = pd.read_csv(
        RESUMES_FILE
    )

    predictions = {}

    print(
        "\nGenerating Rules V2 "
        "predictions..."
    )

    for resume_id in sorted(
        allowed_resume_ids
    ):

        rows = df[
            df["resume_id"]
            == resume_id
        ]

        if rows.empty:

            print(
                f"Warning: {resume_id} "
                f"not found in resumes.csv"
            )

            continue

        text = rows.iloc[0].get(
            "text",
            ""
        )

        if pd.isna(text):
            text = ""

        text = str(text)

        # Existing Rules V2 function.
        regions = generate_annotations(
            text
        )

        entities = set()

        for region in regions:

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

            if label not in LABELS:
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

            entities.add(
                (
                    int(start),
                    int(end),
                    label,
                )
            )

        predictions[
            resume_id
        ] = entities

        print(
            f"  {resume_id}: "
            f"{len(entities)} regions"
        )

    return predictions

# ============================================================
# Hybrid V1 Predictions
# ============================================================

def build_hybrid_predictions(
    rules_predictions,
    llm_predictions,
    allowed_resume_ids
):
    """
    Hybrid V1 based on pilot results.

    Rules V2 performed better for:
        HARD_SKILL
        SOFT_SKILL

    LLM V1 performed better for:
        EXPERIENCE
        EDUCATION
        LANG

    IMPORTANT:
    This strategy was selected using the 5-CV pilot,
    so these CVs are development data, not an unbiased
    final test set.
    """

    RULE_LABELS = {
        "HARD_SKILL",
        "SOFT_SKILL",
    }

    LLM_LABELS = {
        "EXPERIENCE",
        "EDUCATION",
        "LANG",
    }

    predictions = {}

    for resume_id in sorted(
        allowed_resume_ids
    ):

        rules = rules_predictions.get(
            resume_id,
            set()
        )

        llm = llm_predictions.get(
            resume_id,
            set()
        )

        hybrid = set()

        # Rules V2 for HARD_SKILL + SOFT_SKILL
        for entity in rules:

            if entity[2] in RULE_LABELS:

                hybrid.add(
                    entity
                )

        # LLM for EXPERIENCE + EDUCATION + LANG
        for entity in llm:

            if entity[2] in LLM_LABELS:

                hybrid.add(
                    entity
                )

        predictions[
            resume_id
        ] = hybrid

    return predictions

# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    tp,
    fp,
    fn
):

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    ground_truth,
    predictions
):

    counts = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    )

    per_document = []

    common_resume_ids = sorted(
        set(ground_truth)
        & set(predictions)
    )

    for resume_id in common_resume_ids:

        gold = ground_truth[
            resume_id
        ]

        predicted = predictions[
            resume_id
        ]

        # ----------------------------------------------------
        # Per-label counts
        # ----------------------------------------------------

        for label in LABELS:

            gold_label = {
                entity
                for entity in gold
                if entity[2] == label
            }

            pred_label = {
                entity
                for entity in predicted
                if entity[2] == label
            }

            tp = len(
                gold_label
                & pred_label
            )

            fp = len(
                pred_label
                - gold_label
            )

            fn = len(
                gold_label
                - pred_label
            )

            counts[label][
                "tp"
            ] += tp

            counts[label][
                "fp"
            ] += fp

            counts[label][
                "fn"
            ] += fn

        # ----------------------------------------------------
        # Overall per-document counts
        # ----------------------------------------------------

        tp = len(
            gold
            & predicted
        )

        fp = len(
            predicted
            - gold
        )

        fn = len(
            gold
            - predicted
        )

        (
            precision,
            recall,
            f1
        ) = calculate_metrics(
            tp,
            fp,
            fn
        )

        per_document.append(
            {
                "resume_id":
                    resume_id,

                "gold":
                    len(gold),

                "predicted":
                    len(predicted),

                "tp":
                    tp,

                "fp":
                    fp,

                "fn":
                    fn,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,
            }
        )

    return (
        counts,
        per_document,
        common_resume_ids,
    )


# ============================================================
# Print Results
# ============================================================

def print_results(
    method_name,
    counts,
    per_document
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"{method_name} VS HUMAN GROUND TRUTH"
    )

    print(
        "=" * 70
    )

    print(
        "\n"
        f"{'Label':<15}"
        f"{'TP':>7}"
        f"{'FP':>7}"
        f"{'FN':>7}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print(
        "-" * 68
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    label_results = {}

    for label in LABELS:

        tp = counts[
            label
        ]["tp"]

        fp = counts[
            label
        ]["fp"]

        fn = counts[
            label
        ]["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        (
            precision,
            recall,
            f1
        ) = calculate_metrics(
            tp,
            fp,
            fn
        )

        label_results[
            label
        ] = {
            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,
        }

        print(
            f"{label:<15}"
            f"{tp:>7}"
            f"{fp:>7}"
            f"{fn:>7}"
            f"{precision:>12.3f}"
            f"{recall:>10.3f}"
            f"{f1:>10.3f}"
        )

    (
        overall_precision,
        overall_recall,
        overall_f1
    ) = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    print(
        "-" * 68
    )

    print(
        f"{'OVERALL':<15}"
        f"{total_tp:>7}"
        f"{total_fp:>7}"
        f"{total_fn:>7}"
        f"{overall_precision:>12.3f}"
        f"{overall_recall:>10.3f}"
        f"{overall_f1:>10.3f}"
    )

    # ========================================================
    # Per-document
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PER-DOCUMENT RESULTS"
    )

    print(
        "=" * 70
    )

    for item in per_document:

        print(
            f"\n{item['resume_id']}"
        )

        print(
            f"  Human entities: "
            f"{item['gold']}"
        )

        print(
            f"  Predicted:      "
            f"{item['predicted']}"
        )

        print(
            f"  TP: {item['tp']}  "
            f"FP: {item['fp']}  "
            f"FN: {item['fn']}"
        )

        print(
            f"  Precision: "
            f"{item['precision']:.3f}"
        )

        print(
            f"  Recall:    "
            f"{item['recall']:.3f}"
        )

        print(
            f"  F1:        "
            f"{item['f1']:.3f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    mean_document_f1 = (
        sum(
            item["f1"]
            for item in per_document
        )
        / len(per_document)
        if per_document
        else 0.0
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nMethod:          "
        f"{method_name}"
    )

    print(
        f"Micro Precision: "
        f"{overall_precision:.4f}"
    )

    print(
        f"Micro Recall:    "
        f"{overall_recall:.4f}"
    )

    print(
        f"Micro F1:        "
        f"{overall_f1:.4f}"
    )

    print(
        f"Mean document F1:"
        f" {mean_document_f1:.4f}"
    )

    print(
        "\nEvaluation type: "
        "strict entity-level exact match"
    )

    print(
        "A prediction is correct only "
        "when start, end, and label "
        "all match."
    )

    return {
        "precision":
            overall_precision,

        "recall":
            overall_recall,

        "f1":
            overall_f1,

        "mean_document_f1":
            mean_document_f1,

        "labels":
            label_results,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    method = args.method

    print(
        "=" * 70
    )

    print(
        "AUTO-HR PRE-ANNOTATION EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRequested method: "
        f"{method.upper()}"
    )

    # --------------------------------------------------------
    # Human ground truth
    # --------------------------------------------------------

    ground_truth = (
        load_ground_truth()
    )

    print(
        f"Human annotated CVs: "
        f"{len(ground_truth)}"
    )

    pilot_resume_ids = set(
        ground_truth.keys()
    )

    # --------------------------------------------------------
    # Select prediction source
    # --------------------------------------------------------

    if method == "llm":

        predictions = (
            load_llm_predictions(
                allowed_resume_ids=(
                    pilot_resume_ids
                )
            )
        )

        method_name = "LLM V1"

        print(
            f"LLM prediction CVs: "
            f"{len(predictions)}"
        )

    elif method == "rules":

        predictions = (
            load_rules_predictions(
                allowed_resume_ids=(
                    pilot_resume_ids
                )
            )
        )

        method_name = "RULES V2"

        print(
            f"Rules prediction CVs: "
            f"{len(predictions)}"
        )
    elif method == "hybrid":

        print(
            "\nGenerating Rules V2 "
            "component..."
        )

        rules_predictions = (
            load_rules_predictions(
                allowed_resume_ids=(
                    pilot_resume_ids
                )
            )
        )

        print(
            "\nLoading LLM V1 "
            "component..."
        )

        llm_predictions = (
            load_llm_predictions(
                allowed_resume_ids=(
                    pilot_resume_ids
                )
            )
        )

        predictions = (
            build_hybrid_predictions(
                rules_predictions,
                llm_predictions,
                pilot_resume_ids
            )
        )

        method_name = "HYBRID V1"

        print(
            f"Hybrid prediction CVs: "
            f"{len(predictions)}"
        )

    else:

        raise ValueError(
            f"Unsupported method: "
            f"{method}"
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        counts,
        per_document,
        common_resume_ids
    ) = evaluate(
        ground_truth,
        predictions
    )

    print(
        f"CVs evaluated: "
        f"{len(common_resume_ids)}"
    )

    if not common_resume_ids:

        print(
            "\nNo matching resume IDs "
            "were found."
        )

        return

    print_results(
        method_name,
        counts,
        per_document
    )


if __name__ == "__main__":
    main()