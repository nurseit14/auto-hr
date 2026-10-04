# AUTO-HR Resume NER Annotation Guidelines

## Purpose

These guidelines define how resumes must be annotated for the AUTO-HR
Named Entity Recognition dataset.

The dataset contains five entity types:

1. HARD_SKILL
2. SOFT_SKILL
3. EXPERIENCE
4. EDUCATION
5. LANG

All annotators must follow the same rules.

---

# 1. HARD_SKILL

Use HARD_SKILL for specific technical knowledge, technologies,
frameworks, tools, protocols, methodologies, and technical competencies.

## Annotate

Examples:

- Python
- Java
- C++
- SQL
- PostgreSQL
- Docker
- Kubernetes
- Git
- GitHub
- Linux
- Spring Boot
- Spring Security
- Hibernate
- REST API
- FastAPI
- React
- Machine Learning
- Deep Learning
- Cybersecurity
- Wireshark
- TCP/IP
- HTTP
- SIEM
- QRadar
- MITRE ATT&CK
- Figma
- Power BI

Technical concepts may also be annotated when they clearly represent
candidate knowledge or competency.

Examples:

- OOP
- microservices
- network security
- penetration testing
- log analysis
- security monitoring
- IP addressing and subnetting

## Do NOT annotate

Generic terms such as:

- IT
- software
- technology
- programming
- development
- computer
- technical skills

unless the phrase represents a specific technical competency.

Section headings such as:

- TECHNICAL SKILLS
- PROGRAMMING
- TOOLS
- PROJECTS

must not be annotated.

---

# 2. SOFT_SKILL

Use SOFT_SKILL for explicit interpersonal, organizational, or
professional abilities.

## Annotate

Examples:

- teamwork
- leadership
- communication
- analytical thinking
- attention to detail
- problem-solving
- time management
- adaptability
- responsibility
- collaboration

## Do NOT annotate

Generic personality descriptions unless they clearly describe a
professional competency.

Do not annotate section headings such as:

- SOFT SKILLS
- LEADERSHIP & ACTIVITIES

---

# 3. EXPERIENCE

Use EXPERIENCE for actual professional roles, internships, seniority,
or explicit duration of professional experience.

## Annotate

Examples:

- Java Backend Intern
- Data Scientist
- Data Quality Analyst
- Project Manager Assistant
- ML/AI Engineer Intern
- Senior Backend Developer
- Junior Developer
- Team Lead
- 3 years of experience
- 2+ years experience

## Boundary rule

Select the job role only.

Correct:

Java Backend Intern

Incorrect:

Java Backend Intern — Company Name, Astana, 2026

Do not include:

- company name
- city
- employment dates

unless they are part of the actual job title.

## Do NOT annotate

- section heading "EXPERIENCE"
- company names
- dates by themselves
- desired future jobs
- vague phrases such as "some experience"

A professional title in a CV header should not automatically be
annotated as EXPERIENCE unless it represents documented professional
experience.

---

# 4. EDUCATION

Use EDUCATION for degrees, qualifications, and fields of study.

## Annotate

Examples:

- Bachelor of Computer Science
- Bachelor's Degree in Cybersecurity
- B.S. in Computer Science
- Master of Data Science
- Бакалавриат, Software Engineering

Prefer the complete meaningful qualification.

## Do NOT annotate

- university name alone
- school name alone
- GPA
- graduation year
- Grade 9
- education section heading

Example:

Bachelor of Computer Science, Astana IT University

Annotate:

Bachelor of Computer Science

Do not annotate:

Astana IT University

---

# 5. LANG

Use LANG only for human languages.

When proficiency is provided, annotate the language and proficiency
together as ONE entity.

## Correct

- English B2
- English — C1 level
- Russian — Native / Fluent
- Kazakh (native)
- Қазақ тілі — C1

## Incorrect

English        -> LANG
B2             -> separate LANG

These must instead form one entity:

English B2     -> LANG

## Do NOT annotate

Programming languages such as:

- Python
- Java
- C++
- JavaScript

These are HARD_SKILL.

---

# 6. Repeated entities

Annotate every occurrence.

Example:

Python appears three times in a CV.

All three occurrences must be annotated as HARD_SKILL.

Do not annotate only the first occurrence.

---

# 7. Entity boundaries

Always select the smallest complete meaningful entity.

Correct:

Spring Boot

Incorrect:

experience developing applications with Spring Boot

Correct:

English — B2

Incorrect:

English

when B2 is explicitly attached to it.

Avoid accidental partial spans such as:

- "P"
- "ostgreSQL"
- "— Native / Fluent"

Always verify the selected text before submitting.

---

# 8. Certifications

Under the current five-label schema, certifications are NOT assigned a
separate entity label.

Examples:

- Cisco CyberOps Associate
- IBM Machine Learning Certificate
- CCNA

Do not force the entire certification into EDUCATION.

However, a technical skill mentioned inside a certification may still
be annotated when appropriate.

Example:

Cisco Certified Support Technician — Cybersecurity

"Cybersecurity" may be HARD_SKILL.

---

# 9. Projects

Do not annotate project names automatically.

Annotate technical skills appearing inside project descriptions.

Example:

Built a REST API using Python, FastAPI, PostgreSQL and Docker.

Annotate:

- REST API -> HARD_SKILL
- Python -> HARD_SKILL
- FastAPI -> HARD_SKILL
- PostgreSQL -> HARD_SKILL
- Docker -> HARD_SKILL

---

# 10. Pre-annotations

AUTO-HR rules-v2 produces automatic suggestions.

Automatic predictions are NOT ground truth.

For every CV:

1. Review all automatic annotations.
2. Keep correct annotations.
3. Delete incorrect annotations.
4. Fix incorrect boundaries.
5. Add missing entities.
6. Review the complete CV again.
7. Submit only after human verification.

---

# 11. Uncertain cases

If an entity is ambiguous, do not invent a label.

Record the questionable case and discuss it with the team.

After agreement, apply the same decision consistently to all documents.

Consistency is more important than maximizing the number of annotations.

---

# 12. Annotation checklist

Before clicking Submit, verify:

- All obvious HARD_SKILL entities are annotated.
- Explicit SOFT_SKILL entities are annotated.
- Actual job roles / experience durations are annotated.
- Education qualifications are annotated.
- Human languages include proficiency when available.
- Repeated entities are annotated every time.
- No university names are incorrectly labeled EDUCATION.
- No company names or dates are included in EXPERIENCE.
- No section headings are labeled.
- No malformed or partial spans remain.
- All automatic predictions have been manually reviewed.

