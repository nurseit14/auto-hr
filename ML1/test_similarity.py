from ML1.embeddings import encode_texts


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity.

    Our embeddings are already normalized,
    so cosine similarity is simply their dot product.
    """
    return float(vector_a @ vector_b)


def main():

    texts = [
        # 0
        "Python backend developer with Docker and PostgreSQL",

        # 1
        "Backend-разработчик со знанием Python, Docker и PostgreSQL",

        # 2
        "Разработчик интерфейсов React и JavaScript",

        # 3
        "Маркетолог по продвижению товаров и работе с клиентами",

        # 4
        "Python бағдарламашысы, PostgreSQL және Docker тәжірибесі бар"
    ]

    print("Generating embeddings...\n")

    embeddings = encode_texts(texts)

    tests = [
        (0, 1, "English backend vs Russian backend"),
        (0, 2, "Backend vs Frontend"),
        (0, 3, "Backend vs Marketing"),
        (0, 4, "English backend vs Kazakh backend"),
    ]

    print("\nSEMANTIC SIMILARITY RESULTS")
    print("=" * 60)

    for first, second, description in tests:

        score = cosine_similarity(
            embeddings[first],
            embeddings[second]
        )

        print(f"\n{description}")
        print(f"Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
