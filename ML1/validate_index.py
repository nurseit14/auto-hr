"""Check the local index without downloading/loading the embedding model."""
import argparse

from ML1.config import VACANCIES_FILE, EMBEDDINGS_FILE
from ML1.vacancy_index import read_vacancies, load_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        frame, checksum = read_vacancies(VACANCIES_FILE)
        matrix, metadata = load_index(frame, checksum, EMBEDDINGS_FILE)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Invalid vacancy index: {exc}\n")
    print(f"OK: {len(frame)} vacancies, matrix {matrix.shape}, {matrix.dtype}")
    print(f"Model: {metadata['model_name']}")
    print(f"Index version: {metadata['index_version']}")


if __name__ == "__main__":
    main()
