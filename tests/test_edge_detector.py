from pprint import pprint

from src.research.strategy_lab.edge_detector import detect_strategy_edges


sample_rankings = [
    {
        "group": "swing",
        "score": 0.82,
        "metrics": {
            "sample_count": 120,
            "signal_match_rate": 0.62,
            "bias_match_rate": 0.66,
            "avg_future_return_pct": 0.14,
        },
    },
    {
        "group": "intraday",
        "score": 0.71,
        "metrics": {
            "sample_count": 110,
            "signal_match_rate": 0.54,
            "bias_match_rate": 0.58,
            "avg_future_return_pct": 0.07,
        },
    },
    {
        "group": "scalping",
        "score": 0.64,
        "metrics": {
            "sample_count": 130,
            "signal_match_rate": 0.50,
            "bias_match_rate": 0.52,
            "avg_future_return_pct": 0.02,
        },
    },
]


def main():
    result = detect_strategy_edges(sample_rankings, horizon="15m")
    pprint(result)


if __name__ == "__main__":
    main()