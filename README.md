# AUTO-HR — Intelligent Resume–Job Matching System

AUTO-HR is a multilingual NLP and machine learning system for matching IT resumes with job vacancies.

The project focuses on the Kazakhstan IT labor market, where resumes and vacancies may be written in **Kazakh, Russian, English, or a mixture of these languages**.

The current prototype can:

- extract text from PDF resumes;
- anonymize common personal information;
- automatically extract technical skills;
- generate multilingual semantic embeddings;
- compare resumes with job vacancies;
- calculate skill-based and semantic similarity;
- rank vacancies for a given resume;
- return the Top-K most relevant vacancies;
- prepare resumes for Named Entity Recognition (NER) annotation;
- automatically pre-annotate resumes for human verification.

The long-term goal is to develop and evaluate a multilingual resume–vacancy matching pipeline using NER, skill normalization, semantic similarity, ranking, calibration, fairness analysis, and human supervision.

---

# Current Pipeline

The current working prototype follows this pipeline:

```text
                         PDF Resume
                             |
                             v
                      Text Extraction
                             |
                             v
                       Anonymization
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          Skill Extraction        Resume Embedding
                 |                       |
                 |                       |
                 v                       v
          Resume Skills          Semantic Vector
                 |                       |
                 |                       |
                 +-----------+-----------+
                             |
                             v
                     Vacancy Matching
                             |
                  +----------+----------+
                  |                     |
                  v                     v
             Skill Score          Semantic Score
                  |                     |
                  +----------+----------+
                             |
                             v
                       Hybrid Score
                             |
                             v
                   Rank 1,017 Vacancies
                             |
                             v
                       Top-10 Matches
```

---

# Current Dataset

## Job Vacancies

The initial vacancy dataset contained approximately **1,754 job vacancies** from the Kazakhstan labor market.

After preprocessing, filtering, duplicate removal, and cleaning, the current prototype uses:

```text
1,017 cleaned IT vacancies
```

The dataset contains positions such as:

- Backend Developer
- Frontend Developer
- Software Engineer
- Data Analyst
- Data Scientist
- DevOps Engineer
- QA Engineer
- ML Engineer
- IT Specialist
- Programmer

The cleaned vacancy data is used for skill extraction, semantic embedding generation, and resume–vacancy matching.

---

## Resume Dataset

A private resume dataset is currently being collected for research purposes.

Current state:

```text
42 PDF resumes collected
41 resumes with successfully extracted text
```

The resume files and processed resume dataset are **not published in this repository** because they contain private participant data.

The repository contains only the code required to process the data.

---

# Skill Extraction

The first implemented NLP baseline is a dictionary-based skill extractor.

It recognizes technologies and technical competencies such as:

```text
Python
Java
JavaScript
TypeScript
C++
C#
SQL
PostgreSQL
MySQL
MongoDB
Docker
Kubernetes
Git
Linux
FastAPI
Django
Spring Boot
React
PyTorch
TensorFlow
Scikit-learn
Machine Learning
Deep Learning
Data Science
REST API
```

It also contains aliases for different spellings and terminology.

Example:

```text
Postgres
PostgreSQL
Postgre SQL

        ↓

PostgreSQL
```

### Baseline development

The first dictionary version detected at least one skill in:

```text
533 / 1,017 vacancies
Coverage: 52.41%
```

After error analysis and expansion of the dictionary:

```text
722 / 1,017 vacancies
Coverage: 70.99%
```

This dictionary system currently serves as both:

1. a baseline skill extractor;
2. a weak-supervision component for automatic annotation.

---

# Multilingual Semantic Embeddings

AUTO-HR currently uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

to create multilingual semantic embeddings.

Each resume or vacancy is represented as a:

```text
384-dimensional vector
```

All current vacancy embeddings are precomputed and stored as:

```text
1017 × 384
```

This makes resume matching significantly faster because vacancy embeddings do not need to be regenerated for every query.

---

## Cross-Language Test

A simple semantic similarity experiment was performed during prototype development.

Results:

```text
English backend ↔ Russian backend
Similarity: 0.9257

English backend ↔ Kazakh backend
Similarity: 0.7742

Backend ↔ Frontend
Similarity: 0.1336

Backend ↔ Marketing
Similarity: 0.0184
```

These preliminary results indicate that the multilingual embedding model can represent semantically similar job descriptions across different languages.

These values are development tests and should not be interpreted as final research evaluation results.

---

# Resume–Vacancy Matching

AUTO-HR currently combines two signals.

## 1. Skill Similarity

The system extracts skills from the resume and vacancy and measures how much of the detected vacancy skill set is covered by the candidate.

A confidence adjustment is applied when only a small number of vacancy skills were detected.

This prevents a vacancy containing only one detected skill from automatically receiving a misleading 100% skill match.

---

## 2. Semantic Similarity

The complete resume and vacancy texts are converted into multilingual embeddings.

Cosine similarity is then used to calculate:

```text
S_sem
```

This allows resumes and vacancies written in different languages to be compared.

---

## Hybrid Score

The current prototype uses:

```text
S = α × S_skill + (1 - α) × S_sem
```

The current development configuration uses:

```text
α = 0.4
```

therefore:

```text
40% skill similarity
60% semantic similarity
```

This value is currently a **prototype parameter**.

It is not a final experimentally validated value. Future work will tune `α` using validation data.

---

# Example Matching Output

The system can process a real PDF resume and produce output such as:

```text
#1 Data Scientist

Hybrid score:   65.17%
Skill score:    80.00%
Semantic score: 55.29%

Vacancy skills:
data science, machine learning, oracle, python, sql

Matched skills:
data science, machine learning, python, sql

Missing skills:
oracle
```

Other highly ranked positions in the same test included:

```text
Data Engineer
Python Developer
Senior / Lead Data Software Engineer
Python Backend Developer
MLOps Backend Developer
```

---

# PDF Resume Processing

AUTO-HR supports extracting text directly from PDF resumes.

Current pipeline:

```text
PDF
 |
 v
PyPDF
 |
 v
Raw Text
 |
 v
Text Cleaning
 |
 v
Anonymization
 |
 v
Skill Extraction
 |
 v
Semantic Embedding
 |
 v
Job Matching
```

Example:

```bash
python -m ML1.match_cv "path/to/resume.pdf"
```

The system automatically:

1. reads the PDF;
2. extracts text;
3. anonymizes common personal information;
4. detects skills;
5. generates a semantic embedding;
6. compares the resume against all vacancies;
7. returns the Top-10 matching vacancies.

---

# Privacy and Anonymization

Resume data is treated as private research data.

The preprocessing pipeline masks common personal information such as:

```text
email → [EMAIL]
phone → [PHONE]
URL   → [URL]
```

Automatic anonymization is not assumed to be perfect and should be followed by human verification for research data.

Raw and processed resume datasets are excluded from Git.

Example `.gitignore` rules:

```gitignore
data/resumes/raw/
data/resumes/processed/
data/annotations/
```

No participant resumes should be committed to the public repository.

---

# NER Annotation

The research version of AUTO-HR will use Named Entity Recognition to extract structured information from resumes and vacancies.

Five entity types are currently defined:

| Entity | Description | Examples |
|---|---|---|
| `HARD_SKILL` | Technical skills | Python, Docker, SQL |
| `SOFT_SKILL` | Professional/interpersonal skills | teamwork, leadership |
| `EXPERIENCE` | Roles, seniority and experience | Backend Intern, Senior Developer |
| `EDUCATION` | Degree or field of study | Bachelor of Computer Science |
| `LANG` | Human languages and proficiency | English B2 |

---

# Annotation Pilot

A pilot annotation study has been completed using **5 resumes**.

Current pilot statistics:

```text
Documents: 5
Entities: 220
Average entities/document: 44
```

Entity distribution:

```text
HARD_SKILL     178   80.91%
SOFT_SKILL      12    5.45%
EXPERIENCE      12    5.45%
EDUCATION        7    3.18%
LANG            11    5.00%
```

The pilot was used to identify annotation issues such as:

- inconsistent entity boundaries;
- language + proficiency spans;
- experience-role boundaries;
- education boundaries;
- repeated entities;
- certifications;
- generic technical terminology.

The resulting annotation rules are documented in:

```text
NLP/annotation/ANNOTATION_GUIDELINES.md
```

---

# Automatic Pre-Annotation

Manually annotating every resume from scratch is expensive.

AUTO-HR therefore contains an automatic pre-annotation pipeline.

Current version:

```text
autohr-rules-v2
```

It automatically proposes:

```text
HARD_SKILL
SOFT_SKILL
EXPERIENCE
EDUCATION
LANG
```

using:

- skill dictionaries;
- aliases;
- regular expressions;
- pilot-derived terminology.

The generated predictions can be imported into Label Studio.

Human annotators then:

```text
keep correct predictions
        +
delete incorrect predictions
        +
fix incorrect boundaries
        +
add missing entities
        ↓
human-verified ground truth
```

Automatic predictions are **not treated as ground truth** without human verification.

---

# Label Studio

Label Studio is used to create the manually verified NER dataset.

Annotation workflow:

```text
Resume
   ↓
AUTO-HR Rules V2
   ↓
Automatic Pre-annotation
   ↓
Label Studio
   ↓
Human Review
   ↓
Verified Annotation
   ↓
Ground Truth Dataset
```

The current annotation configuration is located in:

```text
NLP/annotation/annotation_config.xml
```

---

# Project Structure

```text
auto-hr/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── resumes/
│   │   ├── raw/                 # private, ignored by Git
│   │   └── processed/           # private, ignored by Git
│   └── annotations/             # private, ignored by Git
│
├── NLP/
│   ├── annotation/
│   │   ├── annotation_config.xml
│   │   ├── annotation_dictionary.py
│   │   ├── ANNOTATION_GUIDELINES.md
│   │   ├── analyze_pilot.py
│   │   ├── prepare_annotation.py
│   │   └── preannotate_resumes.py
│   │
│   ├── anonymizer.py
│   ├── analyze_extraction.py
│   ├── cv_parser.py
│   ├── find_duplicate_resumes.py
│   ├── process_resumes.py
│   ├── process_vacancies.py
│   ├── skill_extractor.py
│   └── skills_dictionary.py
│
├── ML1/
│   ├── embeddings.py
│   ├── build_vacancy_embeddings.py
│   ├── skill_similarity.py
│   ├── matcher.py
│   ├── match_resume.py
│   ├── match_cv.py
│   └── test_similarity.py
│
├── ML2/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Running the Prototype

## Install dependencies

```bash
pip install pandas numpy scikit-learn sentence-transformers pypdf
```

## Process vacancies

```bash
python -m NLP.process_vacancies
```

## Analyze skill extraction

```bash
python -m NLP.analyze_extraction
```

## Build vacancy embeddings

```bash
python -m ML1.build_vacancy_embeddings
```

## Test multilingual similarity

```bash
python -m ML1.test_similarity
```

## Match a PDF CV

```bash
python -m ML1.match_cv "path/to/resume.pdf"
```

---

# Research Roadmap

The current implementation is a **working prototype**, not the final research system.

The next stages are:

```text
Continue collecting resumes
        ↓
Human verification of pre-annotations
        ↓
Inter-annotator agreement
        ↓
Final ground-truth NER dataset
        ↓
Convert annotations to BIO/IOB2
        ↓
Train / validation / test split
        ↓
Compare skill extraction approaches
        ↓
Dictionary baseline
BiLSTM-CRF
mBERT
XLM-RoBERTa
RuBERT
LLM reference
        ↓
Precision / Recall / F1 evaluation
        ↓
Skill normalization
        ↓
Improved resume-job matching
        ↓
Ranking evaluation
        ↓
Calibration
        ↓
Fairness analysis
        ↓
Human-in-the-loop final system
```

---

# Planned NER Model

The current rule-based automatic labeler is only a weak-supervision system.

After enough human-verified annotations are collected, the project will train a real NER model.

The final goal is:

```text
New Resume
    ↓
Trained NER Model
    ↓
HARD_SKILL
SOFT_SKILL
EXPERIENCE
EDUCATION
LANG
    ↓
Skill Normalization
    ↓
Resume–Vacancy Matching
```

This means users of the final system will **not manually annotate their CVs**.

Manual annotation is currently used only to create training and evaluation data.

---

# Current Status

As of the current prototype:

- [x] Vacancy preprocessing
- [x] Vacancy cleaning
- [x] Dictionary skill extraction baseline
- [x] Skill-extraction error analysis
- [x] Multilingual semantic embeddings
- [x] Vacancy embedding index
- [x] Skill similarity
- [x] Semantic similarity
- [x] Hybrid resume–vacancy matching
- [x] PDF resume parsing
- [x] Basic anonymization
- [x] Real PDF → Top-10 vacancy matching
- [x] Resume dataset preprocessing
- [x] Label Studio configuration
- [x] Five-resume annotation pilot
- [x] Annotation guidelines
- [x] Rules-based automatic pre-annotation V1
- [x] Improved automatic pre-annotation V2
- [ ] Complete human-verified resume annotation dataset
- [ ] Inter-annotator agreement evaluation
- [ ] BIO/IOB2 dataset generation
- [ ] Transformer NER training
- [ ] NER model comparison
- [ ] ESCO skill normalization
- [ ] Contrastive matching fine-tuning
- [ ] Ranking evaluation
- [ ] Probability calibration
- [ ] Fairness evaluation
- [ ] Final user interface

---

# Research Goal

The final research system aims to answer three main questions:

1. Which skill-extraction approach performs best for Kazakh, Russian, English, and mixed-language IT recruitment text?
2. Does combining skill-based similarity with multilingual semantic similarity improve resume–vacancy ranking?
3. Can calibrated matching thresholds maintain ranking quality while reducing differences in error rates across candidate groups?

---

# Authors

- Nurseiit Zhuzbay
- Nazerke Kudaybergen
- Kuat Saparali

Astana IT University  
Astana, Kazakhstan

---

## Project Status

**Active development / research prototype**

The current system demonstrates the complete basic flow:

```text
Real PDF CV
    ↓
NLP Processing
    ↓
Skill Extraction
    ↓
Multilingual Semantic Representation
    ↓
Hybrid Matching
    ↓
1,017 Kazakhstan IT Vacancies
    ↓
Top-10 Recommendations
```

The next major milestone is building the human-verified multilingual NER dataset and training the first transformer-based automatic entity extraction model.
