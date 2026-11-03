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

Reuse canonical skill extraction and the trained local ML2 ranker. ML1 embedding artifacts are absent; show lexical/skills fallback honestly and offer optional semantic configuration. Uploaded candidates, explicit candidate-to-vacancy assignments and decisions persist in account-owned SQLite workspaces. Each vacancy compares only its assigned candidates; the general talent pool remains separate. HR can remove an assignment, delete a resume from the workspace, or delete a custom vacancy while retaining its resumes. One login form uses fixed local-demo credentials admin/admin for HR and user/user for candidates, with role checks on the server and no role chooser. Account workspaces persist across logins; resumes stay private to each account. Authenticated candidates see HR-created vacancies in the shared catalogue. These are demo accounts, not verified employer identities. Never automatically reject candidates. No remote key required for the core MVP. Never copy private CVs, annotations, .env or API keys.

Candidates can upload and delete their own resumes, find vacancies, request source-grounded resume feedback and ask about a vacancy or their score. HR has these capabilities plus the existing candidate review workflow and vacancy creation/import. Vacancy import extracts PDF/DOCX/TXT into editable fields before saving. The local helper uses actual vacancy fields, resume evidence and exact score contributions; it must state when information is absent. Questions and generated answers remain separate from CV features and training labels.

ML2 v3 must ground years and seniority in the vacancy's professional profile and technical evidence. Distinguish required skills from explicit wishes and task-grounded skill claims from simple lists. Learn and evaluate candidate-to-vacancy and vacancy-to-candidate ordering separately. Show exact learned contributions and any explicit requirement policy separately. The experience-shortfall rule can reduce a positive score for known insufficient relevant years; unknown years remain unknown. Scores are relative, not probabilities, and extraction is not independent verification of a candidate's claims.

## Brand Commitments

Retain the Auto-HR project name. User requested implementation and integration and delegated routine technical decisions by asking to make all parts work together.

## Evidence on Hand

Existing ML/NLP source; 1,017 public archived vacancies; local trained ML2 model with no candidate text. Assignments 3, 4, 5 and MVP teaching materials supplied. Synthetic demo resumes must be labelled. No invented user testing results.

## Product Principles

Explain evidence and unknowns. Keep every candidate inspectable. Make correction easy. Keep human decisions explicit.

## Open Decisions

User confirmed both HR and candidate workflows and selected visual mockups before code. React/TypeScript + FastAPI are the implementation choice for the existing Python modules. Deployment and production account authentication are outside the requested local branch.
