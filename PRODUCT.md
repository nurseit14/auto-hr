# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Auto-HR serves HR recruiters and technical leads screening IT candidates in Kazakhstan. Assignment 4 establishes the employer workflow. Candidates can provide professional clarifications.

## Product Purpose

Connect the existing NLP, ML1 and ML2 research modules into a usable MVP: vacancy criteria, resumes, relevant clarification, explained comparison, HR correction, shortlist and CSV export. User explicitly requested an HR vacancy publishing workspace as well as candidate-to-vacancy matching, and confirmed strictly IT-sector scope. Final decisions belong to HR.

## Operating Context

Local demonstration and early user testing. Russian interface; source resumes may contain Russian, Kazakh and English. No validated language-specific quality claim. Existing catalogue is a frozen 2023 research dataset, not live job listings.

## Capabilities and Constraints

Reuse canonical skill extraction and the trained local ML2 ranker. ML1 embedding artifacts are absent; show lexical/skills fallback honestly and offer optional semantic configuration. Uploaded candidates and decisions persist in session-scoped SQLite. Local browser sessions are not authenticated employer accounts. Never automatically reject candidates. No remote key required for the core MVP. Never copy private CVs, annotations, .env or API keys.

## Brand Commitments

Retain the Auto-HR project name. User requested implementation and integration and delegated routine technical decisions by asking to make all parts work together.

## Evidence on Hand

Existing ML/NLP source; 1,017 public archived vacancies; local trained ML2 model with no candidate text. Assignments 3, 4, 5 and MVP teaching materials supplied. Synthetic demo resumes must be labelled. No invented user testing results.

## Product Principles

Explain evidence and unknowns. Keep every candidate inspectable. Make correction easy. Keep human decisions explicit.

## Open Decisions

User confirmed both HR and candidate workflows and selected visual mockups before code. React/TypeScript + FastAPI are the implementation choice for the existing Python modules. Deployment and production account authentication are outside the requested local branch.
