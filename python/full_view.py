from typing import Any, Callable, Literal, Self, Sequence, overload
from dataclasses import dataclass
from itertools import product
from enum import StrEnum, auto

import pandas as pd
import numpy as np
from scipy import stats

from python.sampled_hierarchies import SampledHierarchies, SingleStepHierarchies
from python.error_metrics import ErrorMetrics, MetricsIndex as MI
from python.statistical_test import wilcoxon_test, TrendType

type EmFetcher = Callable[[str, str], ErrorMetrics]

def iter_dataset_models():
    return product(Dataset, Model)

def fetch_row_transposed(em: ErrorMetrics, mi: MI):
    return em.get_error(mi).T.values

@dataclass
class HierarchiesErrors:
    sampled_hierarchies: SampledHierarchies
    ems: dict[str, dict[str, ErrorMetrics]]

    def iter_levels(self):
        for h, aggs in self.sampled_hierarchies.hierarchy_level_names.items():
            yield h, ','.join(aggs[1:-1])


try:
    with open('cv/complete_view.csv', 'r') as f:
        COMPLETE_DF = pd.read_csv(f)
except FileNotFoundError:
    COMPLETE_DF = pd.DataFrame()


class Event:
    DEFAULT_ALPHA = 0.05
    _p_values: pd.Series = None
    _desc: str = None
    _reverse: bool = False

    def decision_at(self, alpha: float):
        if not self._reverse:
            return self._p_values.lt(alpha)
        else:
            return self._p_values.ge(alpha)

    @property
    def p_values(self):
        if self._p_values is not None:
            return self._p_values
        raise ValueError(f"P values cannot be retrieved from {self.desc} event")

    @property
    def data(self):
        return self.decision_at(Event.DEFAULT_ALPHA)

    @property
    def desc(self):
        return self._desc

    @staticmethod
    def from_samples(p_values: pd.Series, desc: str = '', reverse: bool = False) -> 'Event':
        e = Event()
        e._p_values = p_values.astype(float)
        e._desc = desc
        e._reverse = reverse
        return e

    def __repr__(self) -> str:
        return self.desc

    def __invert__(self):
        return NotEvent(self)

    def intersection(self, other):
        return self.__and__(other)

    def __and__(self, other: 'Event'):
        if isinstance(other, Event):
            return AndEvent(self, other)

        return NotImplemented

    __rand__ = __and__

    def union(self, other):
        return self.__or__(other)

    def __or__(self, other: 'Event'):
        if isinstance(other, Event):
            return OrEvent(self, other)
        return NotImplemented

    __ror__ = __or__

    def given(self, other: 'Event'):
        return GivenEvent(self, other)

    I = given


class NotEvent(Event):
    def __init__(self, ev: Event):
        self.ev = ev
        self._desc = f'not {ev.desc}'

    def decision_at(self, alpha: float):
        return ~self.ev.decision_at(alpha)


class AndEvent(Event):
    def __init__(self, lhs: Event, rhs: Event) -> None:
        self.lhs = lhs
        self.rhs = rhs

        l_desc = f'({lhs.desc})' if isinstance(self.lhs, (OrEvent, GivenEvent)) else lhs.desc
        r_desc = f'({rhs.desc})' if isinstance(self.rhs, (OrEvent, GivenEvent)) else rhs.desc
        self._desc = f'{l_desc} ∩ {r_desc}'

    def decision_at(self, alpha: float):
        return self.lhs.decision_at(alpha) & self.rhs.decision_at(alpha)


class OrEvent(Event):
    def __init__(self, lhs: Event, rhs: Event) -> None:
        self.lhs = lhs
        self.rhs = rhs

        l_desc = f'({lhs.desc})' if isinstance(self.lhs, (AndEvent, GivenEvent)) else lhs.desc
        r_desc = f'({rhs.desc})' if isinstance(self.rhs, (AndEvent, GivenEvent)) else rhs.desc
        self._desc = f'{l_desc} ∪ {r_desc}'

    def decision_at(self, alpha: float):
        return self.lhs.decision_at(alpha) | self.rhs.decision_at(alpha)


class GivenEvent(Event):
    def __init__(self, lhs: Event, rhs: Event) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self._desc = f'{lhs.desc} | {rhs.desc}'

    def decision_at(self, alpha: float):
        return self.lhs.decision_at(alpha)[self.rhs.decision_at(alpha)]

    @property
    def p_values(self):
        return self.lhs.p_values[self.rhs.data]

class CertainEvent(Event, StrEnum):
    CERTAIN_EVENT = 'Ω'

    def decision_at(self, alpha: float):
        return pd.Series(True, index=COMPLETE_DF.index)

    @property
    def desc(self):
        return str(self)


class ImpossibleEvent(Event, StrEnum):
    IMPOSSIBLE_EVENT = '∅'

    def decision_at(self, alpha: float):
        return pd.Series(False, index=COMPLETE_DF.index)

    @property
    def desc(self):
        return str(self)


class Dataset(Event, StrEnum):
    SIM = 'sim'
    WEB = 'web-traffic'
    M4 = 'M4'
    M5 = 'M5'

    def decision_at(self, alpha: float):
        return COMPLETE_DF['dataset'] == str(self)

    @property
    def desc(self):
        return f"[{self}]"

class Model(Event, StrEnum):
    AUTOETS = 'AutoETS'
    AUTOARIMA = 'AutoARIMA'
    PROPHET = 'Prophet'

    def decision_at(self, alpha: float):
        return COMPLETE_DF['model'] == str(self)

    @property
    def desc(self):
        return f"[{self}]"

class Granularity(StrEnum):
    BOTTOM = 'Bottom'
    TOP = 'Top'

class Transition(StrEnum):
    INCO_TO_RECO = 'inco-to-reco'
    MIN_TO_SINGLE_ROW = 'min-to-single'
    SINGLE_TO_TRIPLE = 'single-to-triple'
    MIN_TO_TRIPLE_ROWS = 'min-to-triple'
    SINGLES_TO_MERGED = 'singles-to-merged'

class Tendency(StrEnum):
    STAGNATING = 'stag'
    DECREASING = 'decr'
    INCREASING = 'incr'

    def __matmul__(self, other: Transition):
        return _TendencyAtTransition(self, other)

class _TendencyAtTransition:
    def __init__(self, ten: Tendency, tra: Transition) -> None:
        self.ten = ten
        self.tra = tra

    def __matmul__(self, other: Granularity):
        tendency, transition, granularity = self.ten, self.tra, other
        reverse = True if tendency == Tendency.STAGNATING else False
        col = f'{tendency}_{transition}_{granularity}'
        data = COMPLETE_DF[col]
        return Event.from_samples(
            data, f'[{tendency} at {transition} at {granularity}]', reverse
        )

class P:
    def __init__(self, samples: Event):
        self.desc = samples.desc
        self.probability = samples.data.sum() / len(samples.data)

    def __float__(self):
        return self.probability

    def __repr__(self) -> str:
        return f'P({self.desc}) = {self.probability:.5f}'

class Corr:
    def __init__(self, samples: Event, other_samples: Event):
        self.desc_1 = str(samples)
        self.desc_2 = str(other_samples)
        self.phi = samples.data.corr(other_samples.data)

    def __float__(self):
        return self.phi

    def __repr__(self) -> str:
        return f'corr({self.desc_1} , {self.desc_2}) = {self.phi:.5f}'


class CombineMethod(StrEnum):
    FISHER = auto()
    STOUFFER = auto()
    PEARSON = auto()
    TIPPETT = auto()
    MUDHOLKAR_GEORGE = auto()


class CombinePValues():
    def __init__(self, method: CombineMethod, samples: Event):
        self.event_desc = samples.desc
        self.method = method
        eps = np.finfo(float).eps
        p = np.clip(samples.p_values, eps, 1.0 - eps)
        _, combined_p = stats.combine_pvalues(p, method=method)
        self.combined_p = combined_p

    def __float__(self):
        return float(self.combined_p)

    def __repr__(self) -> str:
        return f'{self.method} combined p value of {self.event_desc} event is {self.combined_p}'


class EventList(Sequence):
    def __init__(self, *event_samples: Event):
        self.event_samples = event_samples

    def __invert__(self):
        return EventList(*(~e for e in self))

    def __and__(self, other: Event):
        return EventList(*(e & other for e in self))

    __rand__ = __and__

    def __or__(self, other: Event):
        return EventList(*(e | other for e in self))

    __ror__ = __or__

    @overload
    def __mul__(self, other: Event) -> Self: ...
    @overload
    def __mul__(self, other: Self) -> Self: ...
    def __mul__(self, other):
        if isinstance(other, Event):
            return EventList(*(e & o for e in self for o in (~other, other)))
        if isinstance(other, EventList):
            return EventList(*(e & o for e in self for o in other))
        return NotImplemented

    def union(self) -> Event:
        data = self[0]
        for e in self[1:]:
            data |= e
        return data

    def intersection(self) -> Event:
        data = self[0]
        for e in self[1:]:
            data &= e
        return data

    def __iter__(self):
        return iter(self.event_samples)

    def __len__(self):
        return len(self.event_samples)

    def __getitem__(self, index):
        return self.event_samples[index]

    def __repr__(self):
        return repr(self.event_samples)

    def given(self, other):
        if isinstance(other, Event):
            return EventList(*(e.given(other) for e in self))
        if isinstance(other, EventList):
            return EventList(*(e.given(o) for e in self for o in other))
        raise NotImplementedError

    I = given

class CompleteView:
    def __init__(self,
                 single_step_hierarchies: SingleStepHierarchies,
                 merged_sample_hierarchies: SampledHierarchies,
                 single_em_fetcher: EmFetcher,
                 merged_em_fetcher: EmFetcher,
                 avg_em_fetcher: EmFetcher) -> None:
        self.single_hierarchy_bundle = HierarchiesErrors(
            single_step_hierarchies,
            {d: {m: single_em_fetcher(d,m) for m in Model} for d in Dataset}
        )

        self.merged_hierarchy_bundle = HierarchiesErrors(
            merged_sample_hierarchies,
            {d: {m: merged_em_fetcher(d,m) for m in Model} for d in Dataset}
        )

        self.avg_hierarchy_bundle = HierarchiesErrors(
            merged_sample_hierarchies,
            {d: {m: avg_em_fetcher(d,m) for m in Model} for d in Dataset}
        )

        self.make_df_of_errors()
        self.make_df_binary_cols()
        global COMPLETE_DF
        COMPLETE_DF = self._df

    def make_df_of_errors(self):
        dfs = []
        for d, m in iter_dataset_models():
            em = self.single_hierarchy_bundle.ems[d][m]
            df = pd.DataFrame()
            df['ts'] = em.ts_columns
            df['dataset'] = d
            df['model'] = m

            self._init_min_errors(em, df)
            self._init_errors_per_bundle(self.single_hierarchy_bundle, d, m, df)
            self._init_errors_per_bundle(self.merged_hierarchy_bundle, d, m, df)
            self._init_errors_per_bundle(self.avg_hierarchy_bundle, d, m, df, prefix='avg_of_')

            dfs.append(df)
        self._df = pd.concat(dfs, axis=0).reset_index()

    def make_df_binary_cols(self):
        self._binary_cols = []
        for g in Granularity:
            for transition in Transition:
                for tendency in Tendency:
                    col_name = self.run_check(transition, tendency, g, self._df)
                    self._binary_cols.append(col_name)

    def _init_min_errors(self, em, df):
        for g in Granularity:
            df[f'min_inco_{g}'] = fetch_row_transposed(em, MI(hierarchy='Minimal', forecast_type='base_forecasts', granularity=g))
            df[f'min_reco_{g}'] = fetch_row_transposed(em, MI(hierarchy='Minimal', forecast_type='reconciled_forecasts', granularity=g))
        return self

    def min_inco_errors_cols(self, g: Granularity):
        return [f'min_inco_{g}']

    def min_reco_errors_cols(self, g: Granularity):
        return [f'min_reco_{g}']


    def _init_errors_per_bundle(self, bundle: HierarchiesErrors, d, m, df: pd.DataFrame, prefix: str = ''):
        for g in Granularity:
            for h, aggs in bundle.iter_levels():
                df[f'{prefix}{aggs}_{g}'] = fetch_row_transposed(bundle.ems[d][m],
                                                         MI(hierarchy=h, forecast_type='reconciled_forecasts', granularity=g))
        return self

    def single_step_errors_cols(self, g: Granularity):
        return [f'{aggs}_{g}' for _, aggs in self.single_hierarchy_bundle.iter_levels()]

    def merged_sample_errors_cols(self, g: Granularity):
        return [f'{aggs}_{g}' for _, aggs in self.merged_hierarchy_bundle.iter_levels()]


    def _init_average_errors_per_merge(self, d, m, df):
        ''' Unused atm '''
        def agg_to_comp_hier(agg: str):
            for h, levels in self.single_hierarchy_bundle.sampled_hierarchies.hierarchy_level_names.items():
                if agg in levels:
                    return h

        comp_em = self.single_hierarchy_bundle.ems[d][m]
        for g in Granularity:
            for _, aggs in self.merged_hierarchy_bundle.iter_levels():
                comp_hs = [agg_to_comp_hier(a) for a in aggs.split(',')]
                df[f'avg_of_{aggs}_{g}'] = fetch_row_transposed(comp_em, MI(hierarchy=comp_hs, forecast_type='reconciled_forecasts', granularity=g)).mean(axis=1)

    def avg_merged_errors_cols(self, g: Granularity):
        return [f'avg_of_{aggs}_{g}' for _, aggs in self.merged_hierarchy_bundle.iter_levels()]

    def _make_statistical_test_fun(self, base_cols: list[str], new_cols: list[str], tendency: Tendency):
        def statistical_test(row):
            base = row[base_cols].values.flatten().reshape(-1, 1)
            new =  row[new_cols].values.flatten().reshape(-1, 1)
            if base.shape != new.shape: base = np.tile(base, new.shape)
            array = np.hstack((base, new)).astype(float)

            trend_type = {
                Tendency.DECREASING: TrendType.DECREASING,
                Tendency.INCREASING: TrendType.INCREASING,
                Tendency.STAGNATING: TrendType.STAGNANT,
            }[tendency]
            return wilcoxon_test(array, 0.05, trend_type).p_value

        return statistical_test

    def run_check(self, transition: Transition, tendency: Tendency, granularity: Granularity, df: pd.DataFrame) -> str:
        if   transition == Transition.INCO_TO_RECO:
            def do_compare(row):
                cmp_fun = {
                    Tendency.DECREASING: lambda orig,new: orig <= new, # 0 if decreasing -> interpreted as a p value when evaluted
                    Tendency.INCREASING: lambda orig,new: orig >= new,
                    Tendency.STAGNATING: lambda orig,new: np.abs(orig - new) < 1e-12,
                }[tendency]

                return cmp_fun(row[self.min_inco_errors_cols(granularity)[0]], row[self.min_reco_errors_cols(granularity)[0]])

            row_fun = do_compare

        elif transition == Transition.MIN_TO_SINGLE_ROW:
            row_fun = self._make_statistical_test_fun(self.min_reco_errors_cols(granularity),
                                                      self.single_step_errors_cols(granularity),
                                                      tendency)

        elif transition == Transition.SINGLES_TO_MERGED:
            row_fun = self._make_statistical_test_fun(self.avg_merged_errors_cols(granularity),
                                                      self.merged_sample_errors_cols(granularity),
                                                      tendency)

        elif transition == Transition.MIN_TO_TRIPLE_ROWS:
            row_fun = self._make_statistical_test_fun(self.min_reco_errors_cols(granularity),
                                                      self.merged_sample_errors_cols(granularity),
                                                      tendency)

        elif transition == Transition.SINGLE_TO_TRIPLE:
            new_cols  = [c for c in self.merged_sample_errors_cols(granularity) for _ in range(3)]
            base_cols = []
            for _, aggs  in self.merged_hierarchy_bundle.iter_levels():
                individual_aggs = aggs.split(',')
                base_cols.extend([f"{a}_{granularity}" for a in individual_aggs])
            row_fun = self._make_statistical_test_fun(base_cols,
                                                      new_cols,
                                                      tendency)

        col = f'{tendency}_{transition}_{granularity}'
        df[col] = df.apply(row_fun, axis=1)

        return col

