from __future__ import annotations

import json
from typing import Any


class AIPromptBuilder:
    """Build prompt text for higher-level AI interpretation."""

    def build(self, payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, indent=2, ensure_ascii=False)

        return f"""You are a higher-level trading interpreter working on top of a rule-based crypto trading engine.

Your job is not to recalculate indicators from scratch.
Your job is to interpret the structured output from the existing engine and provide a higher-level judgment.

Important rules:
1. Treat the rule-based engine as the primary deterministic layer.
2. Do not override the engine casually.
3. Focus on explaining structure, bottlenecks, and what confirmation is missing.
4. If the engine says no_signal or hold, assess whether that looks correct, slightly conservative, or too strict.
5. Use the provided key_bottlenecks and timeframe summaries to reason efficiently.

Instructions:
1. Read the market summary across all timeframes.
2. Read the rule-based analysis, risk analysis, and execution plan.
3. Explain the current market structure in plain but precise trading language.
4. State whether long, short, or no-trade is currently the most reasonable stance.
5. Explain whether the engine's no_signal/hold stance looks appropriate.
6. Describe what confirmation would be needed next for a valid long setup.
7. Describe what confirmation would be needed next for a valid short setup.
8. End with a concise Telegram-style briefing in 3-5 lines.

Output format:
- Market Structure
- Rule Engine Assessment
- Key Bottlenecks
- Long Scenario
- Short Scenario
- Final Stance
- Telegram Briefing

Structured payload:
{payload_json}
"""