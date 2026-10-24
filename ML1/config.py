"""Shared configuration for vacancy indexing and query embeddings."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VACANCIES_FILE = ROOT_DIR / "data/processed/vacancies_with_skills.csv"
EMBEDDINGS_FILE = ROOT_DIR / "data/processed/vacancy_embeddings.npy"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384
