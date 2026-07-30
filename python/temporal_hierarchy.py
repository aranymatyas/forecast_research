from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from typing import overload
from tqdm import tqdm

from python.utils import long_to_wide, wide_to_long, is_long, get_nearest_monday
from python.aggregation import Aggregation, DAYS, BASE_PERIOD
from python.time_series import TimeSeries, TimeSeriesType
from python.time_series import FULL_DATA, BASE_FORECASTS, TRAIN_SET, VALIDATION_SET, RESIDUALS, RECONCILED_FORECASTS
from python.base_forecasting import make_forecasts

class ForecastLevel:
    def __init__(self, aggregation: Aggregation):
        self.aggregation = aggregation
        self.reset()

    def reset(self):
        self._time_series_library: dict[TimeSeriesType, TimeSeries] = {}
        self._past_reconciled: dict[str, TimeSeries] = {}

    def prepare_aggregation(self, X: pd.DataFrame):
        self[FULL_DATA] = self.aggregation.aggregate_time_series(X)

    def make_forecast(self, h_days: int, model: str, n_jobs: int = -1, verbose: bool = False):
        h = h_days // self.aggregation.period
        freq = f"{self.aggregation.period}D"
        X_train = self[FULL_DATA].wide.head(-h).copy()
        self[TRAIN_SET] = X_train
        self[VALIDATION_SET] = self[FULL_DATA].wide.tail(h)

        forecast_wide, residual_wide = make_forecasts(X_train, model, h, freq, n_jobs, verbose)

        # Reindex to include all original columns (short series become NaN)
        cols = X_train.columns.tolist()
        forecast_wide = forecast_wide.reindex(columns=cols)
        residual_wide = residual_wide.reindex(columns=cols)

        self[BASE_FORECASTS] = (forecast_wide, model)
        self[RESIDUALS] = residual_wide

    @property
    def width_in_flatten(self):
        return BASE_PERIOD // self.aggregation.period

    def process_store_reconciled_forecast(self, reconciled_forecast: np.ndarray, method):
        base_forecasts = self[BASE_FORECASTS].wide
        shape = base_forecasts.values.T.shape
        reconciled_forecasts = reconciled_forecast.reshape(shape).T
        reconciled_forecasts = pd.DataFrame(reconciled_forecasts,
                                            index=base_forecasts.index,
                                            columns=base_forecasts.columns)
        self[RECONCILED_FORECASTS] = (reconciled_forecasts, method)
        self._past_reconciled[method] = self[RECONCILED_FORECASTS] # Allows for multiple reconciliations

    def get_reconciled_forecasts(self, method: str):
        return self._past_reconciled[method]

    def __getitem__(self, type: TimeSeriesType):
        return self._time_series_library.get(type)

    def __setitem__(self, type: TimeSeriesType, data: pd.DataFrame | tuple[pd.DataFrame, str] | TimeSeries):
        if isinstance(data, pd.DataFrame):
            self._time_series_library[type] = TimeSeries(data, self.aggregation)
        elif isinstance(data, tuple):
            self._time_series_library[type] = TimeSeries(data[0], self.aggregation, data[1])
        elif isinstance(data, TimeSeries):
            self._time_series_library[type] = data

    def copy_data(self, from_fl: 'ForecastLevel'):
        self._time_series_library = from_fl._time_series_library.copy()
        self._past_reconciled = from_fl._past_reconciled.copy()
        return self


class TemporalHierarchy:
    def __init__(self, levels: list[Aggregation], name: str = 'th'):
        ''' Levels should be listed top down '''
        if DAYS not in levels:
            print("Days are not included in hierarchy. Are you sure?")
        self.forecast_levels = [ForecastLevel(level) for level in levels]
        self.name = name

    @property
    def aggregations(self):
        return [forecast.aggregation for forecast in self.forecast_levels]


    def make_summing_matrix(self, start: date | None = None, h_days: int = BASE_PERIOD, include_hierarchy_names: bool = True) -> pd.DataFrame:
        ''' Returns the summing matrix defined by the levels of the hierarchy '''
        if start is None:
            start = get_nearest_monday(datetime.now())
        if start.weekday() != 0:
            raise ValueError("Start date must be a monday")
        if h_days % BASE_PERIOD != 0:
            raise ValueError(f"h_days must be divisible by {BASE_PERIOD}")

        base_days = [start + timedelta(days=i) for i in range(h_days)]

        matrices = []
        for level in self.aggregations:
            matrices.append(level.make_summing_matrix(base_days))

        # Post processing
        summing_matrix = pd.concat(matrices, axis=0)
        if not include_hierarchy_names:
            summing_matrix = summing_matrix.drop(columns=['hierarchy'])

        return summing_matrix


    def resample_time_series(self, X: pd.DataFrame):
        ''' Resamples each time series according to the defined levels '''
        if is_long(X):
            X = long_to_wide(X)

        for forecast_lvl in self.forecast_levels:
            if forecast_lvl[FULL_DATA] is None:
                forecast_lvl.prepare_aggregation(X)

        return self

    def make_base_forecasts(self, h_days: int, model: str = 'AutoETS', n_jobs: int = -1, verbose: bool = False):
        ''' Makes base, independent forecasts for each aggregation level. Yields the hierarchy object
        After an aggregation is done
        '''
        if h_days % BASE_PERIOD != 0:
            raise ValueError(f"h_days must be divisible by {BASE_PERIOD}")
        for i, forecast_lvl in enumerate(self.forecast_levels, start=1):
            if forecast_lvl[BASE_FORECASTS] is None:
                if verbose:
                    tqdm.write(f"{forecast_lvl.aggregation} {i}/{len(self.forecast_levels)}")
                forecast_lvl.make_forecast(h_days, model=model, n_jobs=n_jobs, verbose=verbose)
                yield self
        return self

    def store_reconciled_forecasts(self, reconciled_forecasts: np.ndarray, method = None):
        idx = 0
        for forecast_lvl in self.forecast_levels:
            width = forecast_lvl.width_in_flatten
            forecast_at_lvl = reconciled_forecasts[:, :, idx:idx+width]
            forecast_lvl.process_store_reconciled_forecast(forecast_at_lvl, method=method)
            idx += width

    def reorder(self, type: TimeSeriesType):
        ''' Reorder the given data type of each level so that each row is equivalent BASE_PERIOD days.
            Order is from highest level on the left to lowest aggregation on the right '''
        flattened = [forecast_lvl[type].flat for forecast_lvl in self.forecast_levels]
        return np.concat(flattened, axis=2)

    def reset(self):
        ''' Reset all forecast levels '''
        for forecast_lvl in self.forecast_levels:
            forecast_lvl.reset()

    @overload
    def __getitem__(self, key: Aggregation) -> ForecastLevel: ...

    @overload
    def __getitem__(self, key: TimeSeriesType) -> np.ndarray: ...

    def __getitem__(self, key: Aggregation | TimeSeriesType):
        if isinstance(key, Aggregation):
            for lvl in self.forecast_levels:
                if lvl.aggregation == key:
                    return lvl
            raise ValueError(f"Aggregation {key.hierarchy_name} is not part of hierarchy")
        if isinstance(key, TimeSeriesType):
            return self.reorder(key)

    def clone(self, clone_suffix = None, clone_name = None) -> 'TemporalHierarchy':
        ''' Makes a copy of the temporal hierarchy. The copy's hierarchy levels will have the same state and data. '''
        if clone_suffix is not None:
            clone_name = f"{self.name}_{clone_suffix}"
        if clone_name is None:
            raise ValueError("Must specify either clone_suffix or clone_name")
        th = TemporalHierarchy(self.aggregations, clone_name)
        for this, copy in zip(self.forecast_levels, th.forecast_levels):
            copy.copy_data(this)
        return th

    def copy_data(self, from_th: 'TemporalHierarchy'):
        for agg in self.aggregations:
            self[agg].copy_data(from_th[agg])
        return self

    def change_reconciled_forecasts_to(self, method: str):
        ''' Changes the reconciled forecasts to the given method. Those forecasts need to have been calculated. '''
        for forecast_lvl in self.forecast_levels:
            forecast_lvl[RECONCILED_FORECASTS] = forecast_lvl.get_reconciled_forecasts(method)

    def __str__(self):
        return self.name

# Predefined Hierarchies
from python.aggregation import DAYS, WEEKDAYS, WEEKENDS, WEEKS, FORTNIGHTS, MONTHS
from python.aggregation import TWO_DAYS, FOUR_DAYS, ArbitraryAggregation


class MinimalTemporalHierarchy(TemporalHierarchy):
    def __init__(self):
        ''' Minimal temporal hierarchy just with months and days '''
        super().__init__(
            levels=[MONTHS, DAYS],
            name='Minimal'
        )

class SemanticTemporalHierarchy(TemporalHierarchy):
    def __init__(self):
        ''' Temporal hierarchy rough with levels that make semantic sense '''
        super().__init__(
            levels=[MONTHS, FORTNIGHTS, WEEKS, WEEKENDS, WEEKDAYS, DAYS],
            name='Semantic'
        )

class BinaryTemporalHierarchy(TemporalHierarchy):
    def __init__(self):
        ''' Temporal hierarchy roughly binary '''
        super().__init__(
            levels=[MONTHS,
                    ArbitraryAggregation('16Days', [1] * 16 + [0] * 12),
                    ArbitraryAggregation('12Days', [0] * 16 + [1] * 12),
                    ArbitraryAggregation('8Days', [1] * 8 + [0] * 20),
                    ArbitraryAggregation('8Days', [0] * 8 + [1] * 8 + [0] * 12),
                    ArbitraryAggregation('8Days', [0] * 16 + [1] * 8 + [0] * 4),
                    FOUR_DAYS,
                    TWO_DAYS,
                    DAYS],
            name='Binary'
        )

class FullTemporalHierarchy(TemporalHierarchy):
    def __init__(self):
        ''' Full hierarchy with all possible levels '''
        super().__init__(
            levels=[MONTHS, FORTNIGHTS, WEEKS, WEEKENDS, WEEKDAYS,
                    ArbitraryAggregation('16Days', [1] * 16 + [0] * 12),
                    ArbitraryAggregation('12Days', [0] * 16 + [1] * 12),
                    ArbitraryAggregation('8Days', [1] * 8 + [0] * 20),
                    ArbitraryAggregation('8Days', [0] * 8 + [1] * 8 + [0] * 12),
                    ArbitraryAggregation('8Days', [0] * 16 + [1] * 8 + [0] * 4),
                    FOUR_DAYS, TWO_DAYS,
                    DAYS],
            name='Full'
        )
