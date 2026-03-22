from itertools import product
from typing import Literal, Iterable
import pandas as pd
import numpy as np

from python.temporal_hierarchy import TemporalHierarchy
from python.time_series import BASE_FORECASTS, RECONCILED_FORECASTS, VALIDATION_SET, TRAIN_SET, TimeSeriesType
from python.aggregation import DAYS, MONTHS, Aggregation


type MetricType = Literal['mase', 'nrmse', 'wape', '1-r2']
METRICS = ('mase', 'nrmse', 'wape', '1-r2')

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

    def indexes(self):
        return {k: v for k, v in self.dict().items() if not isinstance(v, slice)}

    def __str__(self) -> str:
        return ' / '.join([str(v) for v in self.dict().values() if v != slice(None)])

    @staticmethod
    def indeces_from_product(
        hierarchies: Iterable[TemporalHierarchy | str] | None = None,
        forecast_types: Iterable[TimeSeriesType | str] | None = None,
        granularities: Iterable[Aggregation | str] | None = None,
        error_metrics: Iterable[MetricType] | None = None,
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
        self._cache = True

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]):
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
            epsilon = 1e-8

            for col in self.ts_columns:
                if metric == 'nrmse':
                    error = np.sqrt(np.mean((pred[col] - actual[col])**2)) / (np.std(train[col].dropna()) + epsilon)
                elif metric == 'mase':
                    in_sample_naive_mae = np.mean(np.abs(np.diff(train[col].dropna())))
                    error = np.mean(np.abs(pred[col] - actual[col])) / (in_sample_naive_mae + epsilon)
                elif metric == 'wape':
                    error = np.mean(np.abs(pred[col] - actual[col])) / (np.mean(np.abs(actual[col])) + epsilon)
                elif metric == '1-r2':
                    ss_res = np.sum((actual[col] - pred[col])**2)
                    ss_tot = np.sum((actual[col] - np.mean(actual[col]))**2)
                    if ss_tot < epsilon:
                        error = 0.0 if ss_res < epsilon else np.nan # or a designated penalty value
                    else:
                        error = ss_res / ss_tot
                row[col] = error
            rows.append(row)

        self._metrics = pd.DataFrame(rows).set_index(INDEX_COLS)
        return self

    def calc_ts_columns(self):
        self._ts_columns = self.temporal_hierarchy[DAYS][BASE_FORECASTS].wide.columns.tolist()
        return self

    @property
    def ts_columns(self):
        if self._ts_columns is None or not self._cache:
            self.calc_ts_columns()
        return self._ts_columns

    @property
    def metrics(self):
        if self._metrics is None or not self._cache:
            self.calc_error_metrics()
        return self._metrics

    def drop_cols(self, cols: list[str]):
        self._metrics = self._metrics.drop(columns=cols, errors='ignore')
        self._ts_columns = [col for col in self._ts_columns if col not in cols]
        return self

    def get_error(self, index: MetricsIndex):
        return index.rows(self)

    def get_error_stats(self, index: MetricsIndex = NO_INDEX, df_filter = None):
        df = self.metrics[self.ts_columns]
        if df_filter is not None:
            df = df_filter(df)
        stats = df.agg(['count','mean', 'std', 'min', 'max', 'median',
                                                   lambda x: x.quantile(0.1),
                                                   lambda x: x.quantile(0.25),
                                                   lambda x: x.quantile(0.75),
                                                   lambda x: x.quantile(0.9)],
                                                   axis=1)
        stats.columns = ['count', 'mean', 'std', 'min', 'max', 'median', 'q10', 'q25', 'q75', 'q90']
        stats = index.rows(stats)
        return stats

    @property
    def stats(self):
        return self.get_error_stats()

    def save(self, filename):
        self.metrics.to_csv(filename)


class LoadedErrorMetrics(ErrorMetrics):
    def __init__(self, filename):
        super().__init__(None)
        self._metrics = pd.read_csv(filename, index_col=INDEX_COLS)
        self._ts_columns = self.metrics.columns.tolist()

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]):
        print('Cannot recalculate metrics which were loaded!')
        return self

class ClonedErrorMetrics(ErrorMetrics):
    def __init__(self, error_metrics: ErrorMetrics):
        super().__init__(None)
        self._metrics = error_metrics.metrics.copy()
        self._ts_columns = error_metrics.ts_columns.copy()

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]):
        print('Cannot recalculate metrics which were cloned!')
        return self

class ViewErrorMetrics(ErrorMetrics):
    def __init__(self):
        super().__init__(None)
        self._cache = False # Recalc always as a view

    def calc_ts_columns(self):
        raise NotImplementedError

    def calc_error_metrics(self, _ = None):
        raise NotImplementedError

    def drop_cols(self, cols: list[str]):
        raise NotImplementedError

class MergedErrorMetrics(ViewErrorMetrics):
    def __init__(self, error_metrics_list: list[ErrorMetrics]):
        super().__init__()
        self.error_metrics_list = error_metrics_list

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]):
        if benchmark_aggregations is not None:
            # Called manually
            self._metrics = pd.concat([em.calc_error_metrics(benchmark_aggregations).metrics for em in self.error_metrics_list])
        else:
            # Called by property
            self._metrics = pd.concat([em.metrics for em in self.error_metrics_list])
        return self

    def calc_ts_columns(self):
        ts_cols = [em.ts_columns for em in self.error_metrics_list]
        if not all(cols == ts_cols[0] for cols in ts_cols):
            raise ValueError("All ErrorMetrics must have the same ts_columns")
        self._ts_columns = ts_cols[0]
        return self

    def drop_cols(self, cols: list[str]):
        for em in self.error_metrics_list:
            em.drop_cols(cols)
        return self

    @property
    def metrics(self):
        if self._metrics is None or not self._cache:
            # Explicitly calling with None to signal no recalc of sub metrics
            self.calc_error_metrics(None)
        return self._metrics

class ErrorDifference(ViewErrorMetrics):
    def __init__(self, metrics: ErrorMetrics, index_base: MetricsIndex, index_comp: MetricsIndex):
        super().__init__()
        self.original = metrics
        self.index_base = index_base
        self.index_comp = index_comp
        if index_base.rows(metrics).shape != index_comp.rows(metrics).shape:
            raise ValueError("Metrics cannot be diffed. Indeces must select the same amount of rows")

    def calc_ts_columns(self):
        self._ts_columns = self.original.ts_columns
        return self

    def calc_error_metrics(self, _ = None):
        base = self.index_base.rows(self.original)
        comp = self.index_comp.rows(self.original)

        diff_df = comp.copy()
        diff_df.iloc[:, :] = comp.values - base.values

        comp_idx_df = comp.index.to_frame()
        for idx, filter in self.index_comp.indexes().items():
            base_idx = self.index_base.indexes().get(idx, 'diff')
            comp_idx_df[idx] = f'{filter} - {base_idx}'

        diff_df.index = pd.MultiIndex.from_frame(comp_idx_df)
        self._metrics = diff_df

        return self

    def drop_cols(self, cols: list[str]):
        self.original.drop_cols(cols)

class FilteredErrorMetrics(ViewErrorMetrics):
    def __init__(self, metrics: ErrorMetrics, index: MetricsIndex):
        super().__init__()
        self.original = metrics
        self.index = index

    def calc_ts_columns(self):
        self._ts_columns = self.original.ts_columns
        return self

    def calc_error_metrics(self, _ = None):
        self._metrics = self.index.rows(self.original)
        return self

    def drop_cols(self, cols: list[str]):
        self.original.drop_cols(cols)

class CustomMetrics(ErrorMetrics):
    def __init__(self, metrics, cols):
        super().__init__(None)
        self._metrics = metrics
        self._ts_columns = cols

    def calc_error_metrics(self, benchmark_aggregations: list[Aggregation] = [DAYS, MONTHS]):
        print('Cannont calculate metrics for a CustomMetric')
        return self

    def calc_ts_columns(self):
        print('Cannont calculate columns for a CustomMetric')
        return self

    @classmethod
    def multiply_metrics(cls, em: ErrorMetrics, index_op1: MetricsIndex, index_op2: MetricsIndex):
        ''' Multiplies the specified rows independently per element '''
        op1 = index_op1.rows(em)
        op2 = index_op2.rows(em)

        multiplied_df = op1.copy()
        multiplied_df.iloc[:, :] = op1.values * op2.values

        op1_idx_df = op1.index.to_frame()
        for idx, filter in index_op1.indexes().items():
            op2_idx = index_op2.indexes().get(idx, 'product')
            op1_idx_df[idx] = f'{filter} * {op2_idx}'

        multiplied_df.index = pd.MultiIndex.from_frame(op1_idx_df)
        return CustomMetrics(multiplied_df, em.ts_columns)

    @classmethod
    def keep_outliers(cls, em: ErrorMetrics, threshold: float = 0.95, log = False):
        metrics = pd.DataFrame(index=em.metrics.index)
        cols = None
        for idx, row in em.metrics.iterrows():
            outliers = row[row > row.quantile(threshold)]
            if not cols:
                cols = [f'outlier_{i}' for i in range(len(outliers))]
            metrics.loc[idx, cols] = outliers.values

        if log:
            metrics = np.log2(metrics)

        return CustomMetrics(metrics, cols)
