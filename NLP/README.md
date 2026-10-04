# AUTO-HR NLP Module

The `NLP` module contains the text-processing, skill-extraction, CV-processing, anonymization, and NER annotation components of the AUTO-HR project.

Its main responsibility is to transform unstructured resume and vacancy text into structured information that can later be used by the matching system.

---

## Current NLP Pipeline

### Vacancy Processing

```text
Raw Vacancies
      ↓
Data Cleaning
      ↓
Vacancy Text
      ↓
Dictionary Skill Extraction
      ↓
Normalized Skill Lists
      ↓
vacancies_with_skills.csv
```

### Resume Processing

```text
PDF Resume
     ↓
PDF Text Extraction
     ↓
Text Cleaning
     ↓
Basic Anonymization
     ↓
Skill Extraction
     ↓
Processed Resume
```

### NER Dataset Preparation

```text
Processed Resume
       ↓
Rules V2 Pre-annotation
       ↓
Label Studio
       ↓
Human Review
       ↓
Verified Annotation
       ↓
Future NER Training Dataset
```

---

# Directory Structure

```text
NLP/
│
├── __init__.py
│
├── skills_dictionary.py
├── skill_extractor.py
├── process_vacancies.py
├── analyze_extraction.py
│
├── anonymizer.py
├── cv_parser.py
├── process_resumes.py
├── find_duplicate_resumes.py
│
└── annotation/
    ├── annotation_config.xml
    ├── annotation_dictionary.py
    ├── ANNOTATION_GUIDELINES.md
    ├── prepare_annotation.py
    ├── preannotate_resumes.py
    └── analyze_pilot.py
```

---

# 1. Skill Dictionary

File:

```text
skills_dictionary.py
```

The current baseline uses a manually maintained dictionary of IT skills and aliases.

Example:

```python
"postgresql": [
    "postgresql",
    "postgres",
    "postgre sql"
]
```

This allows different spellings to be mapped to one normalized skill:

```text
Postgres
PostgreSQL
Postgre SQL

     ↓

postgresql
```

The dictionary contains programming languages, frameworks, databases, DevOps technologies, cloud platforms, data/ML technologies, APIs, testing tools, and other technical competencies.

It also includes terminology relevant to the Kazakhstan/CIS labor market, including technologies such as 1C.

The dictionary is a baseline and is not intended to be the final skill-extraction system.

---

# 2. Skill Extractor

File:

```text
skill_extractor.py
```

The skill extractor searches resume or vacancy text for skills defined in the skill dictionary.

Example input:

```text
Backend developer with experience in Python,
FastAPI, PostgreSQL, Docker and Git.
```

Example output:

```text
docker
fastapi
git
postgresql
python
```

The extractor performs basic text normalization and alias matching.

---

# 3. Vacancy Processing

File:

```text
process_vacancies.py
```

This script applies the skill extractor to the cleaned vacancy dataset.

Run:

```bash
python -m NLP.process_vacancies
```

Input:

```text
data/processed/vacancies_clean.csv
```

Output:

```text
data/processed/vacancies_with_skills.csv
```

The output contains additional fields such as:

```text
skills
skill_count
```

---

# 4. Skill Extraction Analysis

File:

```text
analyze_extraction.py
```

This script evaluates the coverage of the dictionary-based extractor.

Run:

```bash
python -m NLP.analyze_extraction
```

It reports:

- total vacancies;
- vacancies with detected skills;
- vacancies without detected skills;
- vacancy-level extraction coverage;
- most common detected skills;
- common words in zero-skill vacancies;
- representative extraction failures.

---

## Dictionary Baseline Results

### Version 1

```text
Total vacancies:          1017
With detected skills:      533
Without detected skills:   484

Coverage: 52.41%
```

After error analysis, the dictionary was expanded using terminology found in the Kazakhstan vacancy dataset.

### Version 2

```text
Total vacancies:          1017
With detected skills:      722
Without detected skills:   295

Coverage: 70.99%
```

The improvement was particularly influenced by technologies and terminology that were common in the local dataset but missing from the original dictionary.

> Coverage is not Precision, Recall, or F1. It only measures the percentage of vacancies for which the baseline detected at least one skill.

Proper NER evaluation will later use manually annotated ground truth.

---

# 5. PDF CV Parser

File:

```text
cv_parser.py
```

The CV parser reads PDF resumes using `pypdf`.

Pipeline:

```text
PDF
 ↓
extract_text_from_pdf()
 ↓
clean_cv_text()
 ↓
anonymize_text()
 ↓
extract_skills()
 ↓
Structured result
```

Example:

```bash
python -m NLP.cv_parser "data/resumes/raw/example.pdf"
```

The parser returns information including:

```text
filename
text
skills
skill_count
```

During prototype testing, 42 PDF resumes were processed:

```text
42 PDF resumes
41 with extractable text
1 requiring future OCR handling
```

---

# 6. Resume Batch Processing

File:

```text
process_resumes.py
```

This script processes all PDF resumes in the private resume directory.

Run:

```bash
python -m NLP.process_resumes
```

Input:

```text
data/resumes/raw/
```

Output:

```text
data/resumes/processed/resumes.csv
```

The generated dataset contains fields such as:

```text
resume_id
filename
text
skills
skill_count
character_count
```

Resume IDs such as:

```text
resume_0001
resume_0002
...
```

are used internally rather than candidate names.

---

# 7. Anonymization

File:

```text
anonymizer.py
```

The baseline anonymizer masks common personally identifying information.

Examples:

```text
email address → [EMAIL]
phone number  → [PHONE]
URL           → [URL]
contact handle → [CONTACT]
```

Automatic anonymization is not assumed to be sufficient for research data.

Human verification is still required.

Raw and processed resumes must not be committed to the public repository.

---

# 8. Duplicate Detection

File:

```text
find_duplicate_resumes.py
```

The initial experiment used semantic embeddings for duplicate detection.

This was found to be inappropriate because different Computer Science resumes can be highly semantically similar.

The current duplicate detector therefore uses:

```text
normalized text
      ↓
SHA-256 exact fingerprint
      +
text sequence similarity
```

This detects document-level duplicates rather than merely semantically similar candidates.

No exact or near-text duplicates were detected among the 41 readable resumes at the current threshold during the latest check.

---

# 9. NER Annotation

The final research system is intended to extract five entity types:

```text
HARD_SKILL
SOFT_SKILL
EXPERIENCE
EDUCATION
LANG
```

Examples:

| Entity | Examples |
|---|---|
| HARD_SKILL | Python, Docker, SQL, Spring Boot |
| SOFT_SKILL | teamwork, leadership, communication |
| EXPERIENCE | Backend Developer Intern, Senior Developer |
| EDUCATION | Bachelor of Computer Science |
| LANG | English B2, Russian C1 |

Annotation is performed using Label Studio.

---

# 10. Annotation Configuration

File:

```text
annotation/annotation_config.xml
```

This defines the five NER labels used in Label Studio.

The annotation rules are documented in:

```text
annotation/ANNOTATION_GUIDELINES.md
```

All annotators should follow the same guidelines.

---

# 11. Annotation Pilot

A five-resume pilot annotation was completed before scaling annotation to the remaining dataset.

Pilot statistics:

```text
Documents: 5
Total entities: 220
Average entities/document: 44
```

Initial entity distribution:

```text
HARD_SKILL     178
SOFT_SKILL      12
EXPERIENCE      12
EDUCATION        7
LANG            11
```

The pilot revealed several annotation issues, including:

- incorrect entity boundaries;
- split language/proficiency spans;
- experience-role boundaries;
- education boundaries;
- treatment of certifications;
- repeated entities;
- generic technical terminology.

These findings were used to refine the annotation guidelines and automatic pre-annotation system.

---

# 12. Automatic Pre-Annotation

File:

```text
annotation/preannotate_resumes.py
```

Manually annotating every entity from scratch is expensive.

AUTO-HR therefore contains a weak-supervision pre-annotation system.

Current version:

```text
autohr-rules-v2
```

It generates automatic suggestions for:

```text
HARD_SKILL
SOFT_SKILL
EXPERIENCE
EDUCATION
LANG
```

using:

```text
main skill dictionary
        +
annotation-specific dictionary
        +
regular expressions
        +
pilot-derived terminology
```

The output can be imported into Label Studio.

---

## Important

Automatic predictions are **not ground truth**.

The workflow is:

```text
Resume
   ↓
Rules V2
   ↓
Automatic predictions
   ↓
Human reviewer
   ├── keeps correct annotations
   ├── removes incorrect annotations
   ├── fixes boundaries
   └── adds missing entities
   ↓
Human-verified annotation
```

The purpose of pre-annotation is to reduce manual work, not replace human verification.

---

# 13. Annotation Dictionary

File:

```text
annotation/annotation_dictionary.py
```

This contains additional terms discovered during the manually annotated pilot.

Examples include technologies and competencies such as:

```text
Spring Security
Hibernate
JPA
JDBC
JWT
BCrypt
Flyway
MapStruct
Maven
Gradle
OOP
Wireshark
Packet Tracer
OSI model
Cybersecurity
LLM
OpenAI API
ETL
```

It also contains pilot-derived soft-skill aliases.

This dictionary is intentionally separated from the primary matching dictionary.

---

# 14. Preparing Label Studio Data

File:

```text
annotation/prepare_annotation.py
```

This converts processed resumes into Label Studio tasks.

Run:

```bash
python -m NLP.annotation.prepare_annotation
```

Example output:

```text
Tasks created: 41
```

---

# 15. Generating Pre-Annotated Tasks

Run:

```bash
python -m NLP.annotation.preannotate_resumes
```

The generated JSON can be imported into a Label Studio project using the configuration in:

```text
annotation/annotation_config.xml
```

Human reviewers must verify every task before it is treated as ground truth.

---

# 16. Pilot Analysis

File:

```text
annotation/analyze_pilot.py
```

This analyzes exported Label Studio JSON.

Run:

```bash
python -m NLP.annotation.analyze_pilot
```

It reports:

- number of annotated documents;
- total entity count;
- distribution by label;
- example entities;
- per-document statistics;
- unexpected labels.

---

# Data Privacy

The following directories contain private or derived participant data and should remain Git-ignored:

```gitignore
data/resumes/raw/
data/resumes/processed/
data/annotations/
```

Do not commit participant CVs to the repository.

---

# Current Status

- [x] Vacancy preprocessing
- [x] Dictionary skill extractor
- [x] Extraction error analysis
- [x] Dictionary V2
- [x] PDF CV parsing
- [x] Basic anonymization
- [x] Resume batch processing
- [x] Text-based duplicate detection
- [x] Label Studio configuration
- [x] Five-CV annotation pilot
- [x] Annotation guidelines
- [x] Rules-based pre-annotation V1
- [x] Rules-based pre-annotation V2
- [ ] Complete human verification of resume annotations
- [ ] Inter-annotator agreement
- [ ] BIO/IOB2 conversion
- [ ] Train NER models
- [ ] Compare NER models
- [ ] Replace dictionary baseline with selected trained model

---

# Planned NER Experiments

After enough human-verified annotations are available, the project plans to compare:

```text
Dictionary baseline
BiLSTM-CRF
Multilingual BERT
XLM-RoBERTa
RuBERT
LLM-based extraction reference
```

Evaluation will use entity-level:

```text
Precision
Recall
F1
```

The best-performing extraction model will eventually replace the rule-based extractor in the production matching pipeline.

---

# Role of This Module

The final responsibility of the NLP module is:

```text
Unstructured Resume / Vacancy
             ↓
        NLP Processing
             ↓
Structured Candidate / Job Information
             ↓
          ML1 Matcher
```

The `ML1` module then uses this structured information together with multilingual semantic representations to calculate resume–vacancy compatibility.
