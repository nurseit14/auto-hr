"""Unified resume CSV loader with schema validation."""

import json
from pathlib import Path

import pandas as pd

from NLP.config import RESUMES_FILE


REQUIRED_COLUMNS = {
    "resume_id",
    "document_id",
    "text_sha256",
    "text",
    "text_length",
    "skills",
}

LEGACY_COLUMNS = {
    "text_hash": "text_sha256",
    "character_count": "text_length",
}


def parse_skills(value):
    """Convert a JSON skills list into a Python list."""

    if isinstance(value, list):
        skills = value
    elif value is None or (
        isinstance(value, float) and pd.isna(value)
    ):
        skills = []
    elif isinstance(value, str):
        if not value.strip():
            skills = []
        else:
            try:
                skills = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid skills JSON: {value!r}"
                ) from error
    else:
        raise ValueError(
            f"Invalid skills value: {value!r}"
        )

    if not isinstance(skills, list):
        raise ValueError(
            "Skills must be a JSON list."
        )

    if not all(isinstance(s, str) for s in skills):
        raise ValueError(
            "Every skill must be a string."
        )

    return skills


def load_resumes(path=None):
    """Load and validate the canonical AUTO-HR dataset."""

    csv_path = Path(path) if path else RESUMES_FILE

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Resume CSV not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        dtype={
            "resume_id": "string",
            "document_id": "string",
            "text_sha256": "string",
            "text_hash": "string",
        },
        keep_default_na=False,
    )

    # Support legacy column names.
    for old_name, new_name in LEGACY_COLUMNS.items():
        if new_name not in df.columns and old_name in df.columns:
            df = df.rename(
                columns={old_name: new_name}
            )

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns in resumes.csv: "
            + ", ".join(missing)
        )

    for column in (
        "resume_id",
        "document_id",
        "text_sha256",
    ):
        if df[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Empty values found in {column}"
            )

    if df["resume_id"].duplicated().any():
        raise ValueError("Duplicate resume_id values.")

    if df["document_id"].duplicated().any():
        raise ValueError("Duplicate document_id values.")

    df["skills"] = df["skills"].apply(parse_skills)

    try:
        df["text_length"] = pd.to_numeric(
            df["text_length"],
            errors="raise",
        ).astype("int64")
    except (ValueError, TypeError) as error:
        raise ValueError(
            "text_length must contain integer values."
        ) from error

    actual_lengths = df["text"].apply(len)

    if not df["text_length"].eq(actual_lengths).all():
        raise ValueError(
            "text_length does not match the actual text length."
        )

    return df
