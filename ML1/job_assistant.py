import os

from dotenv import load_dotenv
from openai import OpenAI

from ML1.assistant_prompt import SYSTEM_PROMPT


load_dotenv()


DEFAULT_MODEL = os.getenv(
    "OPENAI_ASSISTANT_MODEL",
    "gpt-6-luna"
)


# Keep API context reasonably small.
MAX_CV_CHARS = 7000
MAX_VACANCY_CHARS = 8000

# 6 messages = approximately 3 recent user/assistant turns.
MAX_HISTORY_MESSAGES = 6

# Maximum generated answer size.
MAX_OUTPUT_TOKENS = 500


class JobAssistant:

    def __init__(
        self,
        resume_text,
        resume_skills,
        vacancy
    ):

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "OPENAI_API_KEY was not found. "
                "Add it to the .env file."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = DEFAULT_MODEL

        self.resume_text = (
            resume_text or ""
        )

        self.resume_skills = (
            resume_skills or []
        )

        self.vacancy = vacancy

        self.history = []

        self.context = (
            self._build_context()
        )

    # ==================================================
    # Helpers
    # ==================================================

    @staticmethod
    def _clean(value):

        if value is None:
            return ""

        value = str(value).strip()

        if value.lower() == "nan":
            return ""

        return value

    @staticmethod
    def _format_list(values):

        if not values:
            return "None detected"

        if isinstance(values, str):

            if not values.strip():
                return "None detected"

            return values.strip()

        return ", ".join(
            str(value)
            for value in values
        )

    # ==================================================
    # Trusted context
    # ==================================================

    def _build_context(self):

        vacancy = self.vacancy

        cv_text = self.resume_text[
            :MAX_CV_CHARS
        ]

        vacancy_description = (
            self._clean(
                vacancy.get(
                    "description",
                    ""
                )
            )[
                :MAX_VACANCY_CHARS
            ]
        )

        requirements = self._clean(
            vacancy.get(
                "requirements",
                ""
            )
        )

        responsibilities = self._clean(
            vacancy.get(
                "responsibilities",
                ""
            )
        )

        matched = self._format_list(
            vacancy.get(
                "matched_skills",
                []
            )
        )

        missing = self._format_list(
            vacancy.get(
                "missing_skills",
                []
            )
        )

        vacancy_skills = self._format_list(
            vacancy.get(
                "vacancy_skills",
                []
            )
        )

        resume_skills = self._format_list(
            self.resume_skills
        )

        hybrid_score = float(
            vacancy.get(
                "hybrid_score",
                0
            )
        )

        skill_score = float(
            vacancy.get(
                "skill_score",
                0
            )
        )

        semantic_score = float(
            vacancy.get(
                "semantic_score",
                0
            )
        )

        context = f"""
SELECTED VACANCY
================

Title:
{self._clean(vacancy.get("title"))}

City:
{self._clean(vacancy.get("city"))}

Category:
{self._clean(vacancy.get("job_category"))}

Experience requirement:
{self._clean(vacancy.get("experience"))}

Employment:
{self._clean(vacancy.get("employment"))}

Detected vacancy skills:
{vacancy_skills}

Requirements:
{requirements}

Responsibilities:
{responsibilities}

Vacancy text:
{vacancy_description}


AUTO-HR MATCHING ANALYSIS
=========================

IMPORTANT:
These values were calculated by the matching engine.
Use them exactly as supplied.

Hybrid score:
{hybrid_score:.4f}
Displayed percentage:
{hybrid_score:.2%}

Skill score:
{skill_score:.4f}
Displayed percentage:
{skill_score:.2%}

Semantic score:
{semantic_score:.4f}
Displayed percentage:
{semantic_score:.2%}

Current hybrid formula:

hybrid_score =
0.4 * skill_score
+
0.6 * semantic_score

Matched skills:
{matched}

Detected missing skills:
{missing}


CANDIDATE CONTEXT
=================

Detected CV skills:
{resume_skills}

Anonymized CV text:
{cv_text}
""".strip()

        return context

    # ==================================================
    # Terminal display
    # ==================================================

    def show_context(self):

        vacancy = self.vacancy

        print("\n" + "=" * 70)
        print("AUTO-HR LLM ASSISTANT")
        print("=" * 70)

        print(
            f"\nSelected vacancy: "
            f"{vacancy.get('title', 'Unknown')}"
        )

        print(
            f"City: "
            f"{vacancy.get('city', 'Unknown')}"
        )

        print(
            f"Match score: "
            f"{vacancy.get('hybrid_score', 0):.2%}"
        )

        print("\nMatched skills:")

        matched = vacancy.get(
            "matched_skills",
            []
        )

        if matched:

            for skill in matched:
                print(
                    f"  ✓ {skill}"
                )

        else:

            print(
                "  None detected"
            )

        print("\nMissing skills:")

        missing = vacancy.get(
            "missing_skills",
            []
        )

        if missing:

            for skill in missing:
                print(
                    f"  - {skill}"
                )

        else:

            print(
                "  None detected"
            )

        print(
            "\nAsk anything about your CV "
            "and this vacancy."
        )

        print(
            "Type 'exit' to close."
        )

    # ==================================================
    # History
    # ==================================================

    def _history_text(self):

        if not self.history:

            return (
                "No previous conversation."
            )

        messages = []

        for message in self.history[
            -MAX_HISTORY_MESSAGES:
        ]:

            role = (
                message["role"]
                .upper()
            )

            content = message[
                "content"
            ]

            messages.append(
                f"{role}: {content}"
            )

        return "\n".join(
            messages
        )

    # ==================================================
    # Response extraction
    # ==================================================

    @staticmethod
    def _extract_response_text(response):
        """
        Robustly extract text from a Responses API response.
        """

        # Preferred SDK helper.
        output_text = getattr(
            response,
            "output_text",
            ""
        )

        if output_text:

            return str(
                output_text
            ).strip()

        # Fallback: inspect output items.
        text_parts = []

        for item in getattr(
            response,
            "output",
            []
        ):

            for content in getattr(
                item,
                "content",
                []
            ):

                text = getattr(
                    content,
                    "text",
                    None
                )

                if text:

                    text_parts.append(
                        str(text)
                    )

        return "\n".join(
            text_parts
        ).strip()

    # ==================================================
    # API request
    # ==================================================

    def answer(self, question):

        conversation = (
            self._history_text()
        )

        user_input = f"""
TRUSTED AUTO-HR CONTEXT
=======================

{self.context}


RECENT CONVERSATION
===================

{conversation}


CURRENT USER QUESTION
=====================

{question}
""".strip()

        try:

            response = (
                self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=user_input,
                    max_output_tokens=(
                        MAX_OUTPUT_TOKENS
                    ),
                )
            )

        except Exception as error:

            return (
                "The AUTO-HR LLM assistant "
                "could not generate a response. "
                f"API error: {error}"
            )

        answer = (
            self._extract_response_text(
                response
            )
        )

        if not answer:

            answer = (
                "I wasn't able to produce an "
                "answer for that question. "
                "Please try rephrasing it."
            )

        # ----------------------------------------------
        # Save history only after successful request
        # ----------------------------------------------

        self.history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        self.history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        self.history = self.history[
            -MAX_HISTORY_MESSAGES:
        ]

        return answer

    # ==================================================
    # Conversation
    # ==================================================

    def chat(self):

        self.show_context()

        while True:

            try:

                question = input(
                    "\nYou: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError
            ):

                print(
                    "\n\nAUTO-HR Assistant "
                    "closed."
                )

                break

            if question.lower() in {
                "exit",
                "quit",
                "q",
            }:

                print(
                    "\nAUTO-HR: "
                    "Assistant closed."
                )

                break

            if not question:

                continue

            answer = self.answer(
                question
            )

            print(
                f"\nAUTO-HR: {answer}"
            )