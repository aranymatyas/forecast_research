from enum import StrEnum, auto
import pandas as pd

from python.utils import is_long
from python.aggregation import Aggregation, BASE_PERIOD

class TimeSeriesType(StrEnum):
    FULL_DATA = auto()
    TRAIN_SET = auto()
    VALIDATION_SET = auto()
    BASE_FORECASTS = auto()
    RESIDUALS = auto()
    RECONCILED_FORECASTS = auto()

# Aliases
FULL_DATA, TRAIN_SET, VALIDATION_SET, BASE_FORECASTS, RESIDUALS, RECONCILED_FORECASTS = TimeSeriesType


class TimeSeries:
    def __init__(self, data: pd.DataFrame, agg: Aggregation, method: str | None = None):
        if is_long(data):
            raise ValueError("Data must be in wide format")
        self.data = data
        self.width = BASE_PERIOD // agg.period
        self.len = len(data) // self.width
        self._method = method

    @property
    def method(self):
        return self._method or 'NA'

    @property
    def wide(self):
        return self.data

    @property
    def flat(self):
        return self._flatten()

    def _flatten(self):
        return self.data.values.T.reshape(len(self.data.columns), self.len, self.width)
