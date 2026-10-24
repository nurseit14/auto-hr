"""Build the vacancy matrix together with its verifiable manifest."""
from ML1.config import VACANCIES_FILE, EMBEDDINGS_FILE
from ML1.embeddings import encode_texts
from ML1.vacancy_index import read_vacancies, save_index, load_index, manifest_path

OUTPUT_FILE = EMBEDDINGS_FILE


def main():
    print("BUILDING VACANCY EMBEDDINGS")
    frame, csv_sha256 = read_vacancies(VACANCIES_FILE)
    print(f"Loaded {len(frame)} vacancies.")
    embeddings = encode_texts(frame["text"].astype(str).tolist(), batch_size=32)
    # Do not publish an index if its source changed during model inference.
    _, current_sha256 = read_vacancies(VACANCIES_FILE)
    if current_sha256 != csv_sha256:
        raise ValueError("Vacancy CSV changed while building the index. Run the build again.")
    metadata = save_index(frame, csv_sha256, embeddings, OUTPUT_FILE)
    load_index(frame, csv_sha256, OUTPUT_FILE)
    print(f"Saved matrix: {OUTPUT_FILE}")
    print(f"Saved manifest: {manifest_path(OUTPUT_FILE)}")
    print(f"Index version: {metadata['index_version']}")
    print(f"Verified shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
