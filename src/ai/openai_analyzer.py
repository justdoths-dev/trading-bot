from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


class OpenAIAnalyzerError(Exception):
    """Raised when the AI analyzer cannot produce a usable result."""


@dataclass
class OpenAIAnalyzerConfig:
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = 30
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
    environment: str = "paper"


class OpenAIAnalyzer:
    """
    AI interpretation layer.

    Important:
    - AI does NOT replace rule engine.
    - AI interprets rule-based output and risk context.
    """

    def __init__(self, config: OpenAIAnalyzerConfig | None = None) -> None:
        self.config = config or OpenAIAnalyzerConfig()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIAnalyzerError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIAnalyzerError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        self.client = OpenAI(
            api_key=api_key,
            timeout=self.config.timeout_seconds,
        )

    def analyze(self, payload: dict[str, Any], prompt: str) -> dict[str, Any]:
        self._validate_inputs(payload, prompt)

        input_messages = self._build_input(payload, prompt)
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self.client.responses.create(
                    model=self.config.model,
                    input=input_messages,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "trading_interpretation",
                            "strict": True,
                            "schema": self._build_schema(),
                        }
                    },
                )

                response_text = getattr(response, "output_text", None)
                if not response_text:
                    raise OpenAIAnalyzerError("Empty response text from OpenAI.")

                parsed = json.loads(response_text)

                return {
                    "source": "openai",
                    "model": self.config.model,
                    "environment": self.config.environment,
                    "generated_at": int(time.time()),
                    "analysis": parsed,
                }

            except Exception as exc:
                last_error = exc
                if attempt <= self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
                    continue
                break

        return self._build_fallback_result(
            error_message=str(last_error) if last_error else "unknown_error"
        )

    def _validate_inputs(self, payload: dict[str, Any], prompt: str) -> None:
        if not isinstance(payload, dict):
            raise OpenAIAnalyzerError("payload must be a dict.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise OpenAIAnalyzerError("prompt must be a non-empty string.")

    def _build_input(self, payload: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
        system_instruction = (
            "You are a higher-level trading interpreter inside a crypto trading system. "
            "The rule-based engine remains the primary decision-maker. "
            "Do not invent missing market data. "
            "Do not casually override the execution plan. "
            "Interpret structure, bottlenecks, and required confirmations. "
            "Return only valid JSON matching the provided schema."
        )

        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

        return [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nStructured payload:\n{payload_json}",
            },
        ]

    def _build_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "market_structure": {"type": "string"},
                "rule_engine_assessment": {"type": "string"},
                "key_bottlenecks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "long_scenario": {"type": "string"},
                "short_scenario": {"type": "string"},
                "final_stance": {
                    "type": "string",
                    "enum": ["long", "short", "hold"],
                },
                "stance_reason": {"type": "string"},
                "telegram_briefing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 5,
                },
            },
            "required": [
                "market_structure",
                "rule_engine_assessment",
                "key_bottlenecks",
                "long_scenario",
                "short_scenario",
                "final_stance",
                "stance_reason",
                "telegram_briefing",
            ],
        }

    def _build_fallback_result(self, error_message: str) -> dict[str, Any]:
        return {
            "source": "openai_fallback",
            "model": self.config.model,
            "environment": self.config.environment,
            "generated_at": int(time.time()),
            "analysis": {
                "market_structure": "AI interpretation unavailable.",
                "rule_engine_assessment": "Fallback active.",
                "key_bottlenecks": [f"AI unavailable: {error_message}"],
                "long_scenario": "Unavailable.",
                "short_scenario": "Unavailable.",
                "final_stance": "hold",
                "stance_reason": "Fallback mode active.",
                "telegram_briefing": [
                    "AI interpretation unavailable.",
                    "Fallback mode active.",
                    "Hold until system is restored.",
                ],
            },
        }