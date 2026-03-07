from itertools import product
from typing import Literal, Iterable
import pandas as pd
import numpy as np

from python.temporal_hierarchy import TemporalHierarchy
from python.time_series import BASE_FORECASTS, RECONCILED_FORECASTS, VALIDATION_SET, TRAIN_SET, TimeSeriesType
from python.aggregation import DAYS, MONTHS, Aggregation


type MetricType = Literal['mase', 'nrmse', 'nmae']
METRICS = ('mase', 'nrmse', 'nmae')

INDEX_COLS = ['hierarchy', 'forecast_type', 'granularity', 'error_metric', 'method']

class MetricsIndex:
    def __init__(self,
                 hierarchy: list | TemporalHierarchy | str | None = None,
                 forecast_type: list | TimeSeriesType | str | None = None,
                 granularity: list | Aggregation | str | None = None,
                 error_metric: list | MetricType | None = None,
                 method: list | str | None = None
                 ) -> None:
        processed = []
        for arg in [hierarchy, forecast_type, granularity, error_metric, method]:
            if isinstance(arg, (list, tuple)):
                processed.append([str(a) for a in arg])
            elif arg is not None:
                processed.append(str(arg))
            else:
                processed.append(slice(None))

        self.hierarchy, self.forecast_type, self.granularity, self.error_metric, self.method = processed

    def index(self):
        def interval(val):
                    return [val] if not isinstance(val, (list, slice)) else val

        return (
            interval(self.hierarchy),
            interval(self.forecast_type),
            interval(self.granularity),
            interval(self.error_metric),
            interval(self.method)
        )

    def rows(self, metrics_table) -> pd.DataFrame:
        if isinstance(metrics_table, ErrorMetrics):
            metrics_table = metrics_table.metrics
        return metrics_table.loc[self.index()]

    def values(self, metrics_table: 'ErrorMetrics'):
        return self.rows(metrics_table)[metrics_table.ts_columns].values

    def dict(self):
        return {c: v for c, v in zip(INDEX_COLS, (self.hierarchy, self.forecast_type, self.granularity, self.error_metric, self.method))}

    def __str__(self) -> str:
        return ' - '.join([str(v) for v in self.dict().values() if v != slice(None)])

    @staticmethod
    def indeces_from_product(
        hierarchies: Iterable[TemporalHierarchy | str] | None = None,
        forecast_types: Iterable[TimeSeriesType | str] | None = [BASE_FORECASTS, RECONCILED_FORECASTS],
        granularities: Iterable[Aggregation | str] | None = [DAYS, MONTHS],
        error_metrics: Iterable[MetricType] | None = METRICS,
        methods: Iterable[str] | None = None
    ):
        hierarchies = hierarchies or [None]
        forecast_types = forecast_types or [None]
        granularities = granularities or [None]
        error_metrics = error_metrics or [None]
        methods = methods or [None]

        return [
            MetricsIndex(h, f, g, e, m)
            for h, f, g, e, m in product(hierarchies, forecast_types, granularities, error_metrics, methods)
        ]

NO_INDEX = MetricsIndex()


class ErrorMetrics:
    def __init__(self, temporal_hierarchy: TemporalHierarchy):
        self.temporal_hierarchy = temporal_hierarchy
        self._metrics: pd.DataFrame = None
        self._ts_columns: list[str] = None

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]) -> pd.DataFrame:
        ''' Returns table where columns correspond to individual time series and rows are the different error metrics '''
        if not benchmark_aggregations:
            benchmark_aggregations = self.temporal_hierarchy.aggregations
        methods = {
            BASE_FORECASTS: self.temporal_hierarchy[benchmark_aggregations[0]][BASE_FORECASTS].method,
            RECONCILED_FORECASTS: self.temporal_hierarchy[benchmark_aggregations[0]][RECONCILED_FORECASTS].method
        }

        forecasts = [BASE_FORECASTS, RECONCILED_FORECASTS]
        metrics = METRICS
        granularities = benchmark_aggregations

        rows = []
        for forecast_type, granularity, metric in product(forecasts, granularities, metrics):
            index = MetricsIndex(self.temporal_hierarchy, forecast_type, granularity, metric, methods[forecast_type])
            row = index.dict()

            pred = self.temporal_hierarchy[granularity][forecast_type].wide
            actual = self.temporal_hierarchy[granularity][VALIDATION_SET].wide
            train = self.temporal_hierarchy[granularity][TRAIN_SET].wide

            for col in self.ts_columns:
                if metric == 'nrmse':
                    error = np.sqrt(np.mean((pred[col] - actual[col])**2)) / np.std(train[col].dropna())
                elif metric == 'mase':
                    error = np.mean(np.abs(pred[col] - actual[col])) / np.mean(np.abs(np.diff(train[col].dropna())))
                elif metric == 'nmae':
                    error = np.mean(np.abs(pred[col] - actual[col])) / np.mean(train[col].dropna())
                row[col] = error
            rows.append(row)

        self._metrics = pd.DataFrame(rows).set_index(INDEX_COLS)
        return self.metrics

    @property
    def ts_columns(self):
        if self._ts_columns is None:
            self._ts_columns = self.temporal_hierarchy[DAYS][BASE_FORECASTS].wide.columns.tolist()
        return self._ts_columns

    @property
    def metrics(self):
        if self._metrics is None:
            self.calc_error_metrics()
        return self._metrics

    def get_error(self, index: MetricsIndex):
        return index.values(self)

    def get_error_stats(self, index: MetricsIndex = NO_INDEX):
        stats = self.metrics[self.ts_columns].agg(['mean', 'std', 'min', 'max', 'median',
                                                   lambda x: x.quantile(0.1),
                                                   lambda x: x.quantile(0.25),
                                                   lambda x: x.quantile(0.75),
                                                   lambda x: x.quantile(0.9)], axis=1)
        stats.columns = ['mean', 'std', 'min', 'max', 'median', 'q10', 'q25', 'q75', 'q90']
        stats = index.rows(stats)
        return stats

    def save(self, filename):
        self.metrics.to_csv(filename)

class MergedErrorMetrics(ErrorMetrics):
    def __init__(self, error_metrics_list: list[ErrorMetrics]):
        self.error_metrics_list = error_metrics_list
        self._metrics = None
        ts_cols = [em.ts_columns for em in error_metrics_list]
        if not all(cols == ts_cols[0] for cols in ts_cols):
            raise ValueError("All ErrorMetrics must have the same ts_columns")
        self._ts_columns = ts_cols[0]

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]) -> pd.DataFrame:
        self._metrics = pd.concat([em.calc_error_metrics(benchmark_aggregations) for em in self.error_metrics_list])
        return self.metrics

    @property
    def metrics(self):
        if self._metrics is None:
            self.calc_error_metrics()
        return self._metrics

class LoadedErrorMetrics(ErrorMetrics):
    def __init__(self, filename):
        self._metrics = pd.read_csv(filename, index_col=INDEX_COLS)
        self._ts_columns = self.metrics.columns.tolist()

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]) -> pd.DataFrame:
        raise TypeError('Cannot recalculate metrics which were loaded')