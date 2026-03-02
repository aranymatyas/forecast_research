from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from typing import Literal, Optional
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from python.utils import long_to_wide, wide_to_long, is_long, get_nearest_monday
from python.aggregation import Aggregation, DAYS, BASE_PERIOD

class ForecastLevel:
    def __init__(self, aggregation: Aggregation):
        self.aggregation = aggregation
        self.reset()

    def reset(self):
        self.time_series: Optional[pd.DataFrame] = None
        self.base_forecasts: Optional[pd.DataFrame] = None
        self.residuals: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.reconciled_forecasts: Optional[pd.DataFrame] = None

    def prepare_aggregation(self, X: pd.DataFrame):
        self.time_series = self.aggregation.aggregate_time_series(X)

    def make_forecast(self, h_days: int):
        # Remove the validation data
        h = h_days // self.aggregation.period
        X_train = self.time_series.head(-h).copy()
        self.validation_data = self.time_series.tail(h)
        # StatsForecast requires data to be in long format, but conversion is long, so we call it column by column
        sf = StatsForecast(models=[AutoETS()], freq=f"{self.aggregation.period}D", n_jobs=-1)
        X_cols = X_train.columns.tolist()
        forecasts = []
        residuals = []
        for col in X_cols:
            X_train_single = pd.DataFrame({
                'ds': X_train.index.values,
                'y': X_train[col].values,
                'unique_id': 1
            })
            X_train_single = X_train_single.dropna(subset=['y'])
            forecast = sf.forecast(df=X_train_single, h=h, fitted=True)
            forecast = forecast.reset_index(drop=True)
            forecast = forecast[['ds', 'AutoETS']].rename(columns={'AutoETS': col}).set_index('ds')
            forecasts.append(forecast)

            fitted = sf.forecast_fitted_values().reset_index(drop=True)
            fitted['residual'] = fitted['y'] - fitted['AutoETS']
            fitted = fitted[['ds', 'residual']].rename(columns={'residual': col}).set_index('ds')
            residuals.append(fitted)

        self.base_forecasts = pd.concat(forecasts, axis=1)
        self.residuals = pd.concat(residuals, axis=1)

    def flatten_residuals(self):
        return self._flatten(self.residuals)

    def flatten_base_forecasts(self):
        return self._flatten(self.base_forecasts)

    def flatten_validation_data(self):
        return self._flatten(self.validation_data)

    def flatten_reconciled_forecasts(self):
        return self._flatten(self.reconciled_forecasts)

    def _flatten(self, X: pd.DataFrame):
        cols = BASE_PERIOD // self.aggregation.period
        rows = len(X) // cols
        return X.values.T.reshape(len(X.columns), rows, cols)

    @property
    def width_in_flatten(self):
        return BASE_PERIOD // self.aggregation.period

    def store_reconciled_forecast(self, reconciled_forecast: np.ndarray):
        shape = self.base_forecasts.values.T.shape
        reconciled_forecasts = reconciled_forecast.reshape(shape).T
        reconciled_forecasts = pd.DataFrame(reconciled_forecasts, index=self.base_forecasts.index, columns=self.base_forecasts.columns)
        self.reconciled_forecasts = reconciled_forecasts


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
            forecast_lvl.prepare_aggregation(X)

    def make_base_forecasts(self, h_days: int):
        ''' Makes base, independent forecasts for each aggregation level '''
        if h_days % BASE_PERIOD != 0:
            raise ValueError(f"h_days must be divisible by {BASE_PERIOD}")
        for forecast_lvl in self.forecast_levels:
            forecast_lvl.make_forecast(h_days)

    def store_reconciled_forecasts(self, reconciled_forecasts: np.ndarray):
        idx = 0
        for forecast_lvl in self.forecast_levels:
            width = forecast_lvl.width_in_flatten
            forecast_at_lvl = reconciled_forecasts[:, :, idx:idx+width]
            forecast_lvl.store_reconciled_forecast(forecast_at_lvl)
            idx += width

    def reorder_residuals(self):
        ''' Reorder residuals of each level so that each row is equivalent BASE_PERIOD days.
            Order is from highest level on the left to lowest aggregation on the right '''
        return self._concat_flattened(ForecastLevel.flatten_residuals)

    def reorder_base_forecasts(self):
        ''' Reorder base forecasts of each level so that each row is equivalent BASE_PERIOD days.
            Order is from highest level on the left to lowest aggregation on the right '''
        return self._concat_flattened(ForecastLevel.flatten_base_forecasts)

    def reorder_validation_data(self):
        ''' Reorder validation data of each level so that each row is equivalent BASE_PERIOD days.
            Order is from highest level on the left to lowest aggregation on the right '''
        return self._concat_flattened(ForecastLevel.flatten_validation_data)

    def reorder_reconciled_forecasts(self):
        ''' Reorder reconciled forecasts of each level so that each row is equivalent BASE_PERIOD days.
            Order is from highest level on the left to lowest aggregation on the right '''
        return self._concat_flattened(ForecastLevel.flatten_reconciled_forecasts)

    def _concat_flattened(self, flatten_func):
        flattened = [flatten_func(forecast_lvl) for forecast_lvl in self.forecast_levels]
        return np.concat(flattened, axis=2)

    def reset(self):
        for forecast_lvl in self.forecast_levels:
            forecast_lvl.reset()

# Predefined Hierarchies
from python.aggregation import DAYS, WEEKDAYS, WEEKENDS, WEEKS, FORTNIGHTS, MONTHS
class SemanticTemporalHierarchy(TemporalHierarchy):
    def __init__(self):
        super().__init__(
            levels=[MONTHS, FORTNIGHTS, WEEKS, WEEKENDS, WEEKDAYS, DAYS],
            name='Semantic'
        )