from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.notifications.alert_notifier import AlertNotifier
from src.notifications.research_notifier import ResearchNotifier
from src.research.research_metrics import HORIZONS, calculate_research_metrics
from src.research.schema_validator import validate_jsonl_file
from src.research.strategy_lab.comparison_report import (
    compare_by_ai_execution_state,
    compare_by_alignment_state,
    compare_by_strategy,
    compare_by_symbol,
)
from src.research.strategy_lab.dataset_builder import build_dataset
from src.research.strategy_lab.edge_detector import (
    detect_ai_execution_state_edges,
    detect_alignment_state_edges,
    detect_strategy_edges,
    detect_symbol_edges,
)
from src.research.strategy_lab.performance_report import generate_performance_report
from src.research.strategy_lab.ranking_report import (
    rank_by_ai_execution_state,
    rank_by_alignment_state,
    rank_by_strategy,
    rank_by_symbol,
)
from src.research.strategy_lab.segment_report import build_segment_reports

LOGGER = logging.getLogger(__name__)


def load_jsonl_records(input_path: Path) -> list[dict[str, Any]]:
    """Load JSONL records while ignoring blank lines and validating JSON."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            content = line.strip()
            if not content:
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {input_path} at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected JSON object in {input_path} at line {line_number}, got {type(parsed).__name__}"
                )

            records.append(parsed)

    return records


def write_summary_files(
    metrics: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write summary.json and summary.md into output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    summary_json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_md_path.write_text(_build_markdown(metrics), encoding="utf-8")

    return summary_json_path, summary_md_path


def run_research_analyzer(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run full analyzer flow: validate schema, load records, calculate metrics, and write reports."""
    validation_summary = validate_jsonl_file(input_path)

    invalid_records = int(validation_summary.get("invalid_records", 0))
    if invalid_records > 0:
        invalid_examples = validation_summary.get("invalid_examples", [])
        raise ValueError(
            "Schema validation failed before research analysis. "
            f"invalid_records={invalid_records}, invalid_examples={invalid_examples}"
        )

    records = load_jsonl_records(input_path)

    base_metrics = calculate_research_metrics(records)
    strategy_lab_metrics = _build_strategy_lab_metrics(input_path)

    final_metrics = dict(base_metrics)
    final_metrics["schema_validation"] = validation_summary
    final_metrics["strategy_lab"] = strategy_lab_metrics

    write_summary_files(final_metrics, output_dir)
    return final_metrics


def _build_strategy_lab_metrics(input_path: Path) -> dict[str, Any]:
    """Build full Strategy Research Lab metrics bundle."""
    dataset_rows = build_dataset(input_path)

    performance: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    ranking: dict[str, Any] = {}
    edge: dict[str, Any] = {}

    for horizon in HORIZONS:
        performance[horizon] = generate_performance_report(
            horizon=horizon,
            dataset_path=input_path,
        )

        comparison[horizon] = {
            "by_symbol": compare_by_symbol(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_strategy": compare_by_strategy(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_alignment_state": compare_by_alignment_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_ai_execution_state": compare_by_ai_execution_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
        }

        ranking[horizon] = {
            "by_symbol": rank_by_symbol(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_strategy": rank_by_strategy(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_alignment_state": rank_by_alignment_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_ai_execution_state": rank_by_ai_execution_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
        }

        symbol_rankings = _extract_ranking_items(ranking[horizon]["by_symbol"])
        strategy_rankings = _extract_ranking_items(ranking[horizon]["by_strategy"])
        alignment_rankings = _extract_ranking_items(ranking[horizon]["by_alignment_state"])
        ai_execution_rankings = _extract_ranking_items(
            ranking[horizon]["by_ai_execution_state"]
        )

        edge[horizon] = {
            "by_symbol": detect_symbol_edges(
                symbol_rankings,
                horizon=horizon,
            ),
            "by_strategy": detect_strategy_edges(
                strategy_rankings,
                horizon=horizon,
            ),
            "by_alignment_state": detect_alignment_state_edges(
                alignment_rankings,
                horizon=horizon,
            ),
            "by_ai_execution_state": detect_ai_execution_state_edges(
                ai_execution_rankings,
                horizon=horizon,
            ),
        }

    segment = build_segment_reports(
        dataset_rows,
        horizons=tuple(HORIZONS),
        min_samples=10,
    )

    return {
        "dataset_rows": len(dataset_rows),
        "performance": performance,
        "comparison": comparison,
        "ranking": ranking,
        "edge": edge,
        "segment": segment,
    }


def _extract_ranking_items(report: Any) -> list[dict[str, Any]]:
    """Extract ranking rows from ranking report wrapper."""
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]

    if not isinstance(report, dict):
        return []

    items = report.get("rankings")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]

    items = report.get("results")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]

    return []


def _build_markdown(metrics: dict[str, Any]) -> str:
    overview = metrics.get("dataset_overview", {}) or {}
    horizons = metrics.get("horizon_summary", {}) or {}
    by_symbol = metrics.get("by_symbol", {}) or {}
    by_strategy = metrics.get("by_strategy", {}) or {}
    strategy_lab = metrics.get("strategy_lab", {}) or {}
    schema_validation = metrics.get("schema_validation", {}) or {}

    lines: list[str] = []
    lines.append("# Research Summary")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(UTC).isoformat()}")
    lines.append("")

    lines.append("## Schema Validation")
    lines.append("")
    lines.append(f"- total_records: {schema_validation.get('total_records', 0)}")
    lines.append(f"- valid_records: {schema_validation.get('valid_records', 0)}")
    lines.append(f"- invalid_records: {schema_validation.get('invalid_records', 0)}")
    lines.append(f"- error_count: {schema_validation.get('error_count', 0)}")
    lines.append(f"- warning_count: {schema_validation.get('warning_count', 0)}")
    lines.append("")

    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- total_records: {overview.get('total_records', 0)}")
    lines.append(
        f"- records_with_any_future_label: {overview.get('records_with_any_future_label', 0)}"
    )
    lines.append(
        f"- label_coverage_any_horizon_pct: {_fmt_metric(overview.get('label_coverage_any_horizon_pct'))}"
    )

    date_range = overview.get("date_range", {}) or {}
    lines.append(f"- date_range.start: {date_range.get('start', 'unknown')}")
    lines.append(f"- date_range.end: {date_range.get('end', 'unknown')}")
    lines.append("")

    lines.extend(
        _markdown_distribution(
            "symbols_distribution",
            overview.get("symbols_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "selected_strategies_distribution",
            overview.get("selected_strategies_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "bias_distribution",
            overview.get("bias_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "ai_execution_distribution",
            overview.get("ai_execution_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "alignment_distribution",
            overview.get("alignment_distribution", {}),
        )
    )

    lines.append("## Horizon Summary")
    lines.append("")
    for horizon in HORIZONS:
        lines.extend(
            _markdown_horizon_block(
                horizon,
                horizons.get(horizon, {}) or {},
                heading_level=3,
            )
        )

    lines.append("## By Symbol")
    lines.append("")
    if by_symbol:
        for symbol, data in by_symbol.items():
            lines.append(f"### {symbol}")
            lines.append(f"- total_records: {data.get('total_records', 0)}")
            lines.append(
                f"- records_with_any_future_label: {data.get('records_with_any_future_label', 0)}"
            )
            lines.append(
                f"- label_coverage_any_horizon_pct: {_fmt_metric(data.get('label_coverage_any_horizon_pct'))}"
            )
            lines.append("")

            group_horizons = data.get("horizon_summary", {}) or {}
            for horizon in HORIZONS:
                lines.extend(
                    _markdown_horizon_block(
                        horizon,
                        group_horizons.get(horizon, {}) or {},
                        heading_level=4,
                    )
                )
    else:
        lines.append("No symbol groups available.")
        lines.append("")

    lines.append("## By Strategy")
    lines.append("")
    if by_strategy:
        for strategy, data in by_strategy.items():
            lines.append(f"### {strategy}")
            lines.append(f"- total_records: {data.get('total_records', 0)}")
            lines.append(
                f"- records_with_any_future_label: {data.get('records_with_any_future_label', 0)}"
            )
            lines.append(
                f"- label_coverage_any_horizon_pct: {_fmt_metric(data.get('label_coverage_any_horizon_pct'))}"
            )
            lines.append("")

            group_horizons = data.get("horizon_summary", {}) or {}
            for horizon in HORIZONS:
                lines.extend(
                    _markdown_horizon_block(
                        horizon,
                        group_horizons.get(horizon, {}) or {},
                        heading_level=4,
                    )
                )
    else:
        lines.append("No strategy groups available.")
        lines.append("")

    lines.extend(_markdown_strategy_lab_block(strategy_lab))

    return "\n".join(lines).rstrip() + "\n"


def _markdown_strategy_lab_block(strategy_lab: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    lines.append("## Strategy Research Lab")
    lines.append("")
    lines.append(f"- dataset_rows: {strategy_lab.get('dataset_rows', 0)}")
    lines.append("")

    performance = strategy_lab.get("performance", {}) or {}
    ranking = strategy_lab.get("ranking", {}) or {}
    edge = strategy_lab.get("edge", {}) or {}
    segment = strategy_lab.get("segment", {}) or {}

    lines.append("### Performance")
    lines.append("")
    if performance:
        for horizon in HORIZONS:
            report = performance.get(horizon, {}) or {}
            lines.append(f"#### {horizon}")
            lines.append(f"- sample_count: {report.get('sample_count', 0)}")
            lines.append(f"- labeled_count: {report.get('labeled_count', 0)}")
            lines.append(f"- coverage_pct: {_fmt_metric(report.get('coverage_pct'))}")
            lines.append(f"- signal_match_rate: {_fmt_metric(report.get('signal_match_rate'))}")
            lines.append(f"- bias_match_rate: {_fmt_metric(report.get('bias_match_rate'))}")
            lines.append(f"- avg_future_return_pct: {_fmt_metric(report.get('avg_future_return_pct'))}")
            lines.append(f"- median_future_return_pct: {_fmt_metric(report.get('median_future_return_pct'))}")
            lines.append("")
    else:
        lines.append("No performance report available.")
        lines.append("")

    lines.append("### Ranking Highlights")
    lines.append("")
    if ranking:
        for horizon in HORIZONS:
            lines.append(f"#### {horizon}")
            horizon_rank = ranking.get(horizon, {}) or {}

            top_symbol = _extract_top_ranked_group(horizon_rank.get("by_symbol"))
            top_strategy = _extract_top_ranked_group(horizon_rank.get("by_strategy"))
            top_alignment = _extract_top_ranked_group(horizon_rank.get("by_alignment_state"))
            top_ai_execution = _extract_top_ranked_group(horizon_rank.get("by_ai_execution_state"))

            lines.append(f"- top_symbol: {top_symbol}")
            lines.append(f"- top_strategy: {top_strategy}")
            lines.append(f"- top_alignment_state: {top_alignment}")
            lines.append(f"- top_ai_execution_state: {top_ai_execution}")
            lines.append("")
    else:
        lines.append("No ranking report available.")
        lines.append("")

    lines.append("### Edge Highlights")
    lines.append("")
    if edge:
        for horizon in HORIZONS:
            lines.append(f"#### {horizon}")
            horizon_edge = edge.get(horizon, {}) or {}

            lines.append(f"- symbol_edges: {_count_edge_findings(horizon_edge.get('by_symbol'))}")
            lines.append(f"- strategy_edges: {_count_edge_findings(horizon_edge.get('by_strategy'))}")
            lines.append(
                f"- alignment_state_edges: {_count_edge_findings(horizon_edge.get('by_alignment_state'))}"
            )
            lines.append(
                f"- ai_execution_state_edges: {_count_edge_findings(horizon_edge.get('by_ai_execution_state'))}"
            )
            lines.append("")
    else:
        lines.append("No edge report available.")
        lines.append("")

    lines.append("### Segment Highlights")
    lines.append("")
    if segment:
        segment_reports = segment.get("reports", {}) or {}
        for horizon in HORIZONS:
            lines.append(f"#### {horizon}")
            horizon_segments = segment_reports.get(horizon, {}) or {}

            hour_count = (horizon_segments.get("hour_of_day", {}) or {}).get("qualified_segments", 0)
            day_count = (horizon_segments.get("day_of_week", {}) or {}).get("qualified_segments", 0)
            week_part_count = (horizon_segments.get("week_part", {}) or {}).get("qualified_segments", 0)

            lines.append(f"- hour_of_day qualified_segments: {hour_count}")
            lines.append(f"- day_of_week qualified_segments: {day_count}")
            lines.append(f"- week_part qualified_segments: {week_part_count}")
            lines.append("")
    else:
        lines.append("No segment report available.")
        lines.append("")

    return lines


def _extract_top_ranked_group(report: Any) -> str:
    if not isinstance(report, dict):
        return "n/a"

    items = report.get("rankings") or report.get("results") or []
    if not isinstance(items, list) or not items:
        return "n/a"

    first = items[0]
    if not isinstance(first, dict):
        return "n/a"

    return str(first.get("group", "n/a"))


def _count_edge_findings(report: Any) -> int:
    if not isinstance(report, dict):
        return 0
    findings = report.get("edge_findings", [])
    if not isinstance(findings, list):
        return 0
    return len(findings)


def _markdown_horizon_block(
    horizon: str,
    horizon_data: dict[str, Any],
    heading_level: int,
) -> list[str]:
    heading = "#" * heading_level
    lines: list[str] = []
    lines.append(f"{heading} {horizon}")
    lines.append("")
    lines.append(f"- labeled_records: {horizon_data.get('labeled_records', 0)}")

    label_dist = horizon_data.get("label_distribution", {}) or {}
    lines.append(
        "- label_distribution: "
        f"up={label_dist.get('up', 0)}, "
        f"down={label_dist.get('down', 0)}, "
        f"flat={label_dist.get('flat', 0)}"
    )

    lines.append(f"- avg_future_return_pct: {_fmt_metric(horizon_data.get('avg_future_return_pct'))}")
    lines.append(
        f"- median_future_return_pct: {_fmt_metric(horizon_data.get('median_future_return_pct'))}"
    )
    lines.append(f"- positive_rate_pct: {_fmt_metric(horizon_data.get('positive_rate_pct'))}")
    lines.append(f"- negative_rate_pct: {_fmt_metric(horizon_data.get('negative_rate_pct'))}")
    lines.append(f"- flat_rate_pct: {_fmt_metric(horizon_data.get('flat_rate_pct'))}")

    bias_vs_label = horizon_data.get("bias_vs_label", {}) or {}
    lines.append(
        "- bias_vs_label: "
        f"match={bias_vs_label.get('match', 0)}, "
        f"mismatch={bias_vs_label.get('mismatch', 0)}, "
        f"unknown={bias_vs_label.get('unknown', 0)}, "
        f"match_rate_pct={_fmt_metric(bias_vs_label.get('match_rate_pct'))}"
    )

    signal_vs_label = horizon_data.get("signal_vs_label", {}) or {}
    lines.append(
        "- signal_vs_label: "
        f"match={signal_vs_label.get('match', 0)}, "
        f"mismatch={signal_vs_label.get('mismatch', 0)}, "
        f"unknown={signal_vs_label.get('unknown', 0)}, "
        f"match_rate_pct={_fmt_metric(signal_vs_label.get('match_rate_pct'))}"
    )

    lines.append("")
    return lines


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _markdown_distribution(title: str, values: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"### {title}")
    lines.append("")

    if not values:
        lines.append("No data.")
        lines.append("")
        return lines

    for key, count in values.items():
        lines.append(f"- {key}: {count}")

    lines.append("")
    return lines


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "research_reports" / "latest"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trade_analysis.jsonl research metrics")
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory to write summary.json and summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alert_notifier = AlertNotifier()

    try:
        metrics = run_research_analyzer(
            input_path=args.input,
            output_dir=args.output_dir,
        )

        print(
            f"Records analyzed: {metrics.get('dataset_overview', {}).get('total_records', 0)}"
        )
        print(
            f"Strategy lab dataset rows: {metrics.get('strategy_lab', {}).get('dataset_rows', 0)}"
        )
        print(f"Summary JSON: {(args.output_dir / 'summary.json').resolve()}")
        print(f"Summary MD: {(args.output_dir / 'summary.md').resolve()}")

        try:
            notifier = ResearchNotifier()
            notifier.send_latest_summary()
        except Exception:
            LOGGER.exception("ResearchNotifier failed to send summary.")

    except Exception as exc:
        LOGGER.exception("Research analyzer failed.")

        try:
            alert_notifier.send_error_alert(
                source="research_analyzer",
                message="Research analyzer failed",
                details=str(exc),
            )
        except Exception:
            LOGGER.exception("AlertNotifier failed while reporting research analyzer error.")

        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
