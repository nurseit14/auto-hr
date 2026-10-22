"""Shared configuration for the AUTO-HR NLP pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RESUMES_FILE = ROOT_DIR / "data" / "processed" / "resumes.csv"

ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"
