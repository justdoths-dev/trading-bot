from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ai.openai_analyzer import OpenAIAnalyzer, OpenAIAnalyzerConfig
from src.ai.payload_builder import AIPayloadBuilder
from src.ai.prompt_builder import AIPromptBuilder


@dataclass
class AIServiceConfig:
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = 30
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
    environment: str = "paper"
    symbol: str = "BTCUSDT"


class AIService:
    """Orchestrates payload building, prompt building, and OpenAI analysis."""

    def __init__(self, config: AIServiceConfig | None = None) -> None:
        self.config = config or AIServiceConfig()
        self.payload_builder = AIPayloadBuilder(symbol=self.config.symbol)
        self.prompt_builder = AIPromptBuilder()
        self.analyzer = OpenAIAnalyzer(
            config=OpenAIAnalyzerConfig(
                model=self.config.model,
                timeout_seconds=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
                retry_backoff_seconds=self.config.retry_backoff_seconds,
                environment=self.config.environment,
            )
        )

    def run(
        self,
        enriched_data: dict[str, Any],
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
    ) -> dict[str, Any]:
        ai_payload = self.payload_builder.build(
            enriched_data=enriched_data,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
        )

        ai_prompt = self.prompt_builder.build(ai_payload)

        ai_result = self.analyzer.analyze(
            payload=ai_payload,
            prompt=ai_prompt,
        )

        return {
            "payload": ai_payload,
            "prompt": ai_prompt,
            "result": ai_result,
        }