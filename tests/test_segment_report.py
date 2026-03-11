from pprint import pprint

from src.research.strategy_lab.dataset_builder import build_dataset
from src.research.strategy_lab.segment_report import build_segment_reports

sample_rows = [
{
"timestamp": "2026-03-10T12:05:00+00:00",
"symbol": "BTCUSDT",
"selected_strategy": "swing",
"bias": "bullish",
"signal": "long",
"future_label_15m": "up",
"future_return_15m": 0.5,
"future_label_1h": "up",
"future_return_1h": 1.2,
"future_label_4h": "down",
"future_return_4h": -0.3,
},
{
"timestamp": "2026-03-10T03:05:00+00:00",
"symbol": "BTCUSDT",
"selected_strategy": "intraday",
"bias": "bearish",
"signal": "short",
"future_label_15m": "down",
"future_return_15m": -0.2,
"future_label_1h": "down",
"future_return_1h": -0.8,
"future_label_4h": "flat",
"future_return_4h": 0.0,
},
]

reports = build_segment_reports(sample_rows)

pprint(reports)
