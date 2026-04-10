import pandas as pd
import numpy as np
import random
from typing import Callable, Literal

from python.temporal_hierarchy import TemporalHierarchy
from python.aggregation import Aggregation, MONTHS, DAYS
from python.error_metrics import MetricsIndex, ErrorMetrics, MetricsFilter

type StatsMatrix = np.ndarray[tuple[int, int], np.dtype[np.object_]]
type IndexMatrix = np.ndarray[tuple[int, int], np.dtype[np.object_]]

class SampledHierarchies:
    def __init__(self, available_levels: list[Aggregation], d: int, sample_count: int, start_idx: int  = 0):
        self.available_levels = available_levels
        self.d = d
        self._sample_group_ids = range(start_idx, start_idx + sample_count, 1)
        self._heights = range(d, len(available_levels), d)

        self._hierarchies = np.zeros((sample_count, len(self._heights)), dtype=object)

        for i, id in enumerate(self._sample_group_ids):
            level_samples = self.cumulative_samples(id)
            for j, level_sample in enumerate(level_samples):
                th = TemporalHierarchy([MONTHS] + level_sample + [DAYS], f'TH_{id}_{len(level_sample)}')
                self._hierarchies[i, j] = th

        self._full_hierarchy = TemporalHierarchy([MONTHS] + available_levels + [DAYS], 'AllPossible')
        self._base_hierarchy = TemporalHierarchy([MONTHS, DAYS], 'Minimal')


    def cumulative_samples(self, sample_id: int) -> list[list[Aggregation]]:
        ''' Returns n random combinations of levels '''
        rng = random.Random(sample_id)
        chosen = rng.sample(self.available_levels, len(self.available_levels))
        return [chosen[0:n] for n in range(self.d, len(chosen), self.d)]

    @property
    def group_ids(self) -> list[int]:
        return list(self._sample_group_ids)

    @property
    def heights(self) -> list[int]:
        return list(self._heights)

    @property
    def hierarchies(self) -> list[TemporalHierarchy]:
        return list(self._hierarchies.flatten())

    @property
    def full_hierarchy(self) -> TemporalHierarchy:
        return self._full_hierarchy

    @full_hierarchy.setter
    def full_hierarchy(self, value: TemporalHierarchy):
        self._full_hierarchy = value

    @property
    def base_hierarchy(self) -> TemporalHierarchy:
        return self._base_hierarchy

    @base_hierarchy.setter
    def base_hierarchy(self, value: TemporalHierarchy):
        self._base_hierarchy = value

    @property
    def hierarchy_level_names(self) -> dict[str, list[str]]:
        return {h.name: [str(a) for a in h.aggregations] for h in self.hierarchies}

    def indeces(self, group: int | None = None, height: int | None = None, include_ends: bool = False, **index_kwargs) -> list[MetricsIndex]:
        return self._indeces_helper(group, height, include_ends, index_kwargs, str)

    def index_matrix(self, **index_kwargs):
        return np.vectorize(lambda h: MetricsIndex(**index_kwargs | {'hierarchy': h}))(self._hierarchies)

    def diff_indeces(self, group: int | None = None, height: int | None = None, include_ends: bool = False, **index_kwargs):
        return self._indeces_helper(group, height, include_ends, index_kwargs, lambda h: f'{h} - {self.base_hierarchy}')

    def diff_index_matrix(self, **index_kwargs):
        return np.vectorize(lambda h: MetricsIndex(**index_kwargs | {'hierarchy': f'{h} - {self.base_hierarchy}'}))(self._hierarchies)

    def _indeces_helper(self, group: int | None, height: int | None, include_ends: bool, index_kwargs: dict,
                        name_func: Callable[[TemporalHierarchy], str]) -> list[MetricsIndex]:
        if group is not None:
            group = self._sample_group_ids.index(group)
            hiers = list(self._hierarchies[group, :])
        elif height is not None:
            height = self._heights.index(height)
            hiers = list(self._hierarchies[:, height])
        else:
            raise ValueError('Either group or height needs to be specified')

        if include_ends:
                hiers = [self.base_hierarchy] + hiers + [self.full_hierarchy]

        return [MetricsIndex(**(index_kwargs | {'hierarchy': name_func(h) })) for h in hiers]

    def aggregate_over(self, group_by: Literal['groups', 'heights'],
                       stats: list[str], aggs: list[str], *,
                       df_filter: MetricsFilter=None,
                       stats_matrix: StatsMatrix | None=None, em: ErrorMetrics | None = None, index_matrix: IndexMatrix | None=None) -> 'MetricAggregateStore':
        if stats_matrix is None and (em is None or self.index_matrix is None):
            raise ValueError('Either stats_matrix or both em and index_matrix need to be specified')

        axis = 1 if group_by == 'groups' else 0
        all_stats = self.error_matrix(em, index_matrix, df_filter) if stats_matrix is None else stats_matrix
        agg_store = MetricAggregateStore()
        for stat in stats:
            for agg in aggs:
                agg_store[stat, agg, group_by] = getattr(np, agg)(np.vectorize(lambda s: s[stat].item())(all_stats), axis=axis)
        return agg_store

    def error_matrix(self, em: ErrorMetrics, index_matrix: IndexMatrix, df_filter: MetricsFilter=None) -> np.ndarray:
        return np.vectorize(lambda i: em.get_error_stats(i, df_filter), otypes=[object])(index_matrix)

    def error_row(self, em: ErrorMetrics, indeces: list[MetricsIndex], df_filter: MetricsFilter=None):
        return np.vectorize(lambda i: em.get_error_stats(i, df_filter), otypes=[object])(indeces)

    def trend_check(self, stats_matrix: StatsMatrix, stat: str, trend: Literal['increase', 'decrease']) -> float:
        per_group = self.trend_check_per_group(stats_matrix, stat, trend)
        return np.mean(per_group)

    def trend_check_per_group(self, stats_matrix: StatsMatrix, stat: str, trend: Literal['increase', 'decrease']) -> np.ndarray:
        values = np.vectorize(lambda s: s[stat].item())(stats_matrix)
        desired_change = 1 if trend == 'increase' else 0
        changes = np.where(np.diff(values, axis=1) > 0, desired_change, 1 - desired_change)

        return np.mean(changes, axis=1)

class MetricAggregate:
    def __init__(self, stat: str, agg: str, values: np.ndarray, group_by: str):
        self.stat = stat
        self.agg = agg
        self.values = values
        self.group_by = group_by

    def __str__(self) -> str:
        return f'{self.agg} aggregate of {self.stat} statistic'

class MetricAggregateStore:
    def __init__(self):
        self._data: dict[str, dict[str, MetricAggregate]] = {}

    @property
    def stats(self) -> list[str]:
        return list(self._data.keys())

    @property
    def aggs(self) -> list[str]:
        return [list(a.keys())[0] for a in self._data.values()]

    def __setitem__(self, key, value):
        stat, agg, group_by = key
        if stat not in self._data:
            self._data[stat] = {}
        self._data[stat][agg] = MetricAggregate(stat, agg, value, group_by)

    def __getitem__(self, key: tuple[str, str]) -> MetricAggregate:
        stat, agg = key
        return self._data[stat][agg]

    def __iter__(self):
        for s, agg_value in self._data.items():
            for a, _ in agg_value.items():
                yield s, a
