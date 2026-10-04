# AUTO-HR ML1 — Resume–Vacancy Matching

The `ML1` module contains the semantic representation and resume–vacancy matching components of AUTO-HR.

Its primary responsibility is to answer:

> How well does a candidate resume match a particular job vacancy?

The current prototype combines:

1. skill-based similarity;
2. multilingual semantic similarity.

These signals are combined into a hybrid matching score and used to rank vacancies.

---

# Current Architecture

```text
                         Resume
                           |
              +------------+------------+
              |                         |
              v                         v
        Skill Extraction          Text Embedding
              |                         |
              v                         v
        Resume Skills             Resume Vector
              |                         |
              |                         |
              v                         v
        Vacancy Skills         Vacancy Embedding
              |                         |
              v                         v
           S_skill                   S_sem
              |                         |
              +------------+------------+
                           |
                           v
                    Hybrid Score
                           |
                           v
                    Vacancy Ranking
                           |
                           v
                       Top-K Jobs
```

---

# Directory Structure

```text
ML1/
│
├── __init__.py
├── embeddings.py
├── test_similarity.py
├── build_vacancy_embeddings.py
├── skill_similarity.py
├── matcher.py
├── match_resume.py
└── match_cv.py
```

---

# 1. Multilingual Embeddings

File:

```text
embeddings.py
```

The current prototype uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The model converts text into a:

```text
384-dimensional embedding vector
```

Example:

```text
"Python backend developer with Docker"

              ↓

Multilingual Transformer

              ↓

[0.024, -0.081, ..., 0.042]

        384 dimensions
```

Embeddings are normalized during generation.

Because they are normalized, cosine similarity can be calculated efficiently using the vector dot product.

---

# 2. Multilingual Similarity Test

File:

```text
test_similarity.py
```

Run:

```bash
python -m ML1.test_similarity
```

A prototype cross-language test produced:

```text
English backend vs Russian backend
Similarity: 0.9257

English backend vs Kazakh backend
Similarity: 0.7742

Backend vs Frontend
Similarity: 0.1336

Backend vs Marketing
Similarity: 0.0184
```

These development results suggest that the pretrained multilingual model can place semantically similar descriptions from different languages close together.

They are not final research evaluation results.

---

# 3. Vacancy Embedding Index

File:

```text
build_vacancy_embeddings.py
```

Generating embeddings for every vacancy every time a CV is submitted would be inefficient.

Therefore vacancy embeddings are generated once and stored on disk.

Run:

```bash
python -m ML1.build_vacancy_embeddings
```

Current index:

```text
1,017 vacancies
       ↓
Multilingual MiniLM
       ↓
1,017 × 384 embedding matrix
```

Output:

```text
data/processed/vacancy_embeddings.npy
```

This allows new resumes to be compared against all vacancies without regenerating the vacancy representations.

---

# 4. Semantic Similarity

For a resume `R` and vacancy `V`, the semantic component is conceptually:

```text
S_sem(R,V) = cosine(
    embedding(R),
    embedding(V)
)
```

Because the current embeddings are normalized, the implementation uses a dot product.

The semantic model uses the complete resume and vacancy text rather than only extracted keywords.

This helps capture relationships such as:

```text
"developed REST services using Python"

             ↕

"backend API development experience"
```

even when the exact wording differs.

---

# 5. Skill Similarity

File:

```text
skill_similarity.py
```

The skill component compares skills extracted by the NLP module.

Example:

```text
Resume:

Python
Docker
PostgreSQL
Git


Vacancy:

Python
Docker
PostgreSQL
Kubernetes
```

Matched:

```text
Python
Docker
PostgreSQL
```

Missing:

```text
Kubernetes
```

---

## Confidence Adjustment

A problem was discovered in the first matching prototype.

If the vacancy extractor detected only:

```text
Python
```

and the resume contained Python, a simple coverage calculation produced:

```text
1 / 1 = 100%
```

This was misleading because the vacancy probably contained additional requirements that the dictionary failed to extract.

The current prototype therefore applies a confidence adjustment based on the number of detected vacancy skills.

Conceptually:

```text
coverage =
matched_skills / vacancy_skills

confidence =
min(1, number_of_vacancy_skills / target)

adjusted_skill_score =
coverage × confidence
```

This reduces the influence of poorly represented vacancies.

---

# 6. Hybrid Matching Score

File:

```text
matcher.py
```

The prototype combines skill and semantic similarity:

```text
S = α × S_skill + (1 - α) × S_sem
```

Current prototype setting:

```text
α = 0.4
```

Therefore:

```text
40% skill similarity
60% semantic similarity
```

The semantic component currently receives more weight because the full-text embedding representation is richer than the dictionary-based vacancy skill representation.

This value is a **development setting**, not a final
