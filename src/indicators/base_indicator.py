"""Base types and shared validation for indicators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseIndicator(ABC):
    """Abstract interface for technical indicators."""

    required_columns: tuple[str, ...] = ("close",)

    @classmethod
    def validate_input(cls, data: pd.DataFrame) -> None:
        """Validate DataFrame input for required indicator columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Indicator input must be a pandas DataFrame.")

        missing = [column for column in cls.required_columns if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # numeric validation for close column
        if not pd.api.types.is_numeric_dtype(data["close"]):
            raise TypeError("'close' column must be numeric.")

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Return indicator values aligned to the input DataFrame index."""