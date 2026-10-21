"""The automatic annotation model used by ML2."""

import json
import os
import uuid
from pathlib import Path

from openai import OpenAI

from ML2.data import digest
from ML2.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from ML2.schemas import (PairAssessment, ModelPairAssessment, ground_model_answer,
                         numbered_lines, validate_assessment)

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ML2Annotator:
    def __init__(self, model: str = DEFAULT_MODEL, reference_date: str = "2026-10-21"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing; set it in the project .env")
        self.model = model
        self.reference_date = reference_date
        self.client = OpenAI(timeout=120, max_retries=2)
        self.version = f"ml2-{PROMPT_VERSION}-{model}"

    def annotate(self, pair: dict, cache_dir: Path) -> dict:
        payload = {"reference_date": self.reference_date,
                   "cv": {"lines": [{"line_id": line["line_id"], "text": line["text"]}
                                    for line in numbered_lines(pair["cv"]["text"])]},
                   "vacancy": {**pair["vacancy"], "lines": [
                       {"line_id": line["line_id"], "text": line["text"]}
                       for line in numbered_lines(pair["vacancy"]["display"])]}}
        fingerprint = digest({"input": payload, "model": self.model, "prompt": SYSTEM_PROMPT,
                              "schema": ModelPairAssessment.model_json_schema()})
        path = cache_dir / (fingerprint + ".json")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            answer = PairAssessment.model_validate(record["assessment"])
            validate_assessment(answer, pair["cv"]["text"], pair["vacancy"])
            return {**record, "from_cache": True}
        validation_error = ""
        for attempt in range(3):
            instructions = SYSTEM_PROMPT
            if validation_error:
                instructions += "\nПредыдущий ответ не прошёл проверку: " + validation_error
                instructions += "\nИсправь ответ. Цитаты копируй дословно из исходного JSON."
            response = self.client.responses.parse(
                model=self.model, instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False), text_format=ModelPairAssessment,
                max_output_tokens=10000, store=False,
            )
            raw_answer = response.output_parsed
            if raw_answer is None:
                raise ValueError("Model refused or returned an incomplete structured answer")
            try:
                answer = ground_model_answer(raw_answer, pair["cv"]["text"], pair["vacancy"]["display"])
                validate_assessment(answer, pair["cv"]["text"], pair["vacancy"])
            except ValueError as error:
                validation_error = str(error)
                if attempt == 2:
                    raise
                continue
            record = {"assessment": answer.model_dump(), "model": self.model,
                      "model_version": self.version, "prompt_version": PROMPT_VERSION,
                      "response_id": response.id, "cache_key": fingerprint,
                      "usage": response.usage.model_dump() if response.usage else {},
                      "reference_date": self.reference_date, "from_cache": False}
            save_json(path, record)
            return record
        raise RuntimeError("Model validation did not complete")
