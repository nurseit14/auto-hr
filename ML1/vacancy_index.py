"""Fail-closed validation of the CSV, matrix and index manifest as one artifact."""
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ML1.config import EMBEDDING_DIMENSION, MODEL_NAME

REBUILD = " Rebuild with: python -m ML1.build_vacancy_embeddings"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def manifest_path(matrix_path):
    return Path(matrix_path).with_suffix(".manifest.json")


def validate_vacancy_ids(frame):
    if "vacancy_id" not in frame.columns:
        raise ValueError(
            "Vacancy dataset must contain a 'vacancy_id' column. "
            "Row numbers and the legacy 'id' column are not vacancy IDs."
        )
    ids = frame["vacancy_id"].tolist()
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("Vacancy dataset contains empty vacancy_id values.")
    if any(value != value.strip() for value in ids):
        raise ValueError("Vacancy IDs must not contain leading or trailing whitespace.")
    if len(ids) != len(set(ids)):
        raise ValueError("Vacancy dataset contains duplicate vacancy_id values.")


def read_vacancies(path):
    # Parse the exact bytes we hash; a concurrent source edit cannot change the snapshot.
    data = Path(path).read_bytes()
    frame = pd.read_csv(io.BytesIO(data), dtype={"vacancy_id": str, "text": str}, keep_default_na=False)
    validate_vacancy_ids(frame)
    if "text" not in frame.columns:
        raise ValueError("Column 'text' not found in vacancy dataset.")
    if frame.empty:
        raise ValueError("Vacancy dataset is empty.")
    return frame, sha256(data)


def row_identity(frame):
    return [
        {"vacancy_id": vid, "text_sha256": sha256(str(text).encode("utf-8"))}
        for vid, text in zip(frame["vacancy_id"], frame["text"])
    ]


def validate_matrix(matrix, row_count):
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Vacancy embeddings must be a two-dimensional matrix." + REBUILD)
    if matrix.shape[0] != row_count:
        raise ValueError("Vacancy dataset and embeddings have different sizes." + REBUILD)
    if matrix.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Vacancy embedding dimension does not match the configured model." + REBUILD)
    if matrix.dtype.kind != "f" or not np.isfinite(matrix).all():
        raise ValueError("Vacancy embeddings must contain finite floating-point values." + REBUILD)
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("Vacancy embeddings must be L2-normalized." + REBUILD)


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def index_version(metadata):
    return sha256(json.dumps(
        {k: v for k, v in metadata.items() if k not in {"created_at", "index_version"}},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8"))


def save_index(frame, csv_sha256, matrix, path):
    validate_matrix(matrix, len(frame))
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    data = buffer.getvalue()
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": csv_sha256,
        "matrix_sha256": sha256(data),
        "model_name": MODEL_NAME,
        "normalize_embeddings": True,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "rows": row_identity(frame),
    }
    metadata["index_version"] = index_version(metadata)
    # Publish manifest last. An interrupted or concurrent write fails checksum validation.
    _atomic_write(path, data)
    _atomic_write(manifest_path(path), (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return metadata


def load_index(frame, csv_sha256, path):
    try:
        metadata = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Vacancy index manifest is missing or invalid." + REBUILD) from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported vacancy index manifest schema." + REBUILD)
    if metadata.get("model_name") != MODEL_NAME or metadata.get("normalize_embeddings") is not True:
        raise ValueError("Vacancy index model/settings do not match current configuration." + REBUILD)
    if metadata.get("csv_sha256") != csv_sha256 or metadata.get("rows") != row_identity(frame):
        raise ValueError("Vacancy CSV content or row order has changed since indexing." + REBUILD)
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("Vacancy embedding matrix is missing or unreadable." + REBUILD) from exc
    if metadata.get("matrix_sha256") != sha256(data):
        raise ValueError("Vacancy embedding matrix checksum mismatch." + REBUILD)
    try:
        matrix = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError("Invalid vacancy embedding matrix." + REBUILD) from exc
    validate_matrix(matrix, len(frame))
    if metadata.get("shape") != list(matrix.shape) or metadata.get("dtype") != str(matrix.dtype):
        raise ValueError("Vacancy index shape/dtype metadata mismatch." + REBUILD)
    if metadata.get("index_version") != index_version(metadata):
        raise ValueError("Vacancy index version fingerprint mismatch." + REBUILD)
    return matrix, metadata
