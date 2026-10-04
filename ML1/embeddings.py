from sentence_transformers import SentenceTransformer


# Multilingual embedding model
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def get_model():
    """
    Load the embedding model only once.
    """

    global _model

    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")

        _model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded.")

    return _model


def encode_text(text):
    """
    Convert one text into a normalized embedding vector.
    """

    model = get_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding


def encode_texts(texts, batch_size=32):
    """
    Convert multiple texts into normalized embedding vectors.
    """

    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":

    print("Testing multilingual embeddings...\n")

    texts = [
        "Python backend developer with Docker and PostgreSQL",
        "Backend-разработчик со знанием Python, Docker и PostgreSQL",
        "Маркетолог по продвижению товаров"
    ]

    embeddings = encode_texts(texts)

    print()
    print("Embedding shape:")
    print(embeddings.shape)

    print()
    print("First 10 values of first embedding:")
    print(embeddings[0][:10])
