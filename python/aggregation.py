from datetime import date
import pandas as pd
import numpy as np

from python.utils import is_long

BASE_PERIOD = 28

class Aggregation:
    def __init__(self, hierarchy_name: str, period: int, density: float):
        if BASE_PERIOD % period != 0:
            raise ValueError(f"Interval + skip must be divisor of {BASE_PERIOD}")
        self.hierarchy_name = hierarchy_name
        self.period = period
        self.density = density

    def make_summing_matrix(self, base_days: list[date]) -> pd.DataFrame:
        raise NotImplementedError

    def aggregate_time_series(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._aggregate(X)

    def _aggregate(self, X: pd.DataFrame, frequency: str | None = None) -> pd.DataFrame:
        if is_long(X):
            raise ValueError("X must be in wide format for aggregation")
        if frequency:
            # If neat frequency can be given, i.e. skip is zero
            return X.resample(frequency, closed='right', label='right', origin='end').sum(min_count=1)
        else:
            # Filter out skip days, then aggregate
            X = X.copy()
            X['day_in_period'] = np.arange(len(X)) % self.period
            X['period_id'] = np.arange(len(X)) // self.period
            X = self._select_keep_days(X)

            period_end_dates = X.index.to_series().groupby(X['period_id']).last()

            X_agg = X.groupby('period_id').sum(min_count=1)
            X_agg.index = period_end_dates.loc[X_agg.index]
            X_agg.index.name = X.index.name

            return X_agg.drop(columns=['day_in_period'])

    def _select_keep_days(self, X: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def from_summing_matrix(summing_matrix: np.ndarray, hierarchy_name: str = 'Arbitrary') -> list['Aggregation']:
        return [ArbitraryAggregation(hierarchy_name, row) for row in summing_matrix]

    def __str__(self):
        return self.hierarchy_name

class ArbitraryAggregation(Aggregation):
    def __init__(self, hierarchy_name: str, summing_row: np.ndarray | list):
        summing_row = np.array(summing_row, dtype=np.int32)
        super().__init__(hierarchy_name, BASE_PERIOD, summing_row.sum() / BASE_PERIOD)
        if (BASE_PERIOD, ) != summing_row.shape:
            raise ValueError(f'Summing row must be full length of {BASE_PERIOD}')
        if not (set(summing_row) <= {0, 1}):
            raise ValueError('Summing row must only contain 0s and 1s')
        self.summing_row = summing_row

    def make_summing_matrix(self, base_days: list[date]) -> pd.DataFrame:
        df = pd.DataFrame([self.summing_row], index=[base_days[-1]], columns=base_days)
        df['hierarchy'] = self.hierarchy_name
        return df

    def _select_keep_days(self, X: pd.DataFrame):
        return X[X['day_in_period'].isin(np.flatnonzero(self.summing_row))]

class PeriodicAggregation(Aggregation):

    def __init__(self, hierarchy_name: str, interval: int, skip: int = 0, left: bool = True):
        super().__init__(hierarchy_name, interval + skip, interval / (interval + skip))
        self.left = left # interval on the left and skip on right
        self.interval = interval
        self.skip = skip

    def make_summing_matrix(self, base_days: list[date]) -> pd.DataFrame:
        h_days = len(base_days)
        end_index = self.interval - 1 if self.left else self.period - 1
        day_indexes = [base_days[i+end_index] for i in range(0, h_days, self.period)]

        matrix = np.zeros((h_days // self.period, h_days), dtype=int)
        for i in range(0, h_days, self.period):
            start = i if self.left else i + self.skip
            end = start + self.interval
            matrix[i // self.period, start:end] = 1
        df = pd.DataFrame(matrix, index=day_indexes, columns=base_days)
        df['hierarchy'] = self.hierarchy_name

        return df

    def aggregate_time_series(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' Aggregates time series according to the defined level. X must be in wide format at this point. '''
        if self.skip == 0:
            return self._aggregate(X, f'{self.interval}D')
        else:
            return self._aggregate(X)

    def _select_keep_days(self, X: pd.DataFrame):
        if self.left:
            return X[X['day_in_period'] < self.interval]
        else:
            return X[X['day_in_period'] >= self.skip]

# Predefined aggregations for easier access
DAYS = PeriodicAggregation('Days', 1, 0, True)
WEEKDAYS = PeriodicAggregation('Weekdays', 5, 2, True)
WEEKENDS = PeriodicAggregation('Weekends', 2, 5, False)
WEEKS = PeriodicAggregation('Weeks', 7, 0, True)
FORTNIGHTS = PeriodicAggregation('Fortnights', 14, 0, True)
MONTHS = PeriodicAggregation('Months', 28, 0, True)

TWO_DAYS = PeriodicAggregation('2Days', 2, 0, True)
FOUR_DAYS = PeriodicAggregation('4Days', 4, 0, True)
