from dataclasses import dataclass, field
from typing import Callable, Literal
from python.sampled_hierarchies import SampledHierarchies, StatsMatrix, SingleStepHierarchies
from python.error_metrics import ErrorMetrics, MetricsIndex
from python.time_series import RECONCILED_FORECASTS
from python.aggregation import Aggregation, TOP_LEVEL, BOTTOM_LEVEL

import numpy as np
import pandas as pd
from scipy import stats
from enum import StrEnum, auto

class TrendType(StrEnum):
    DECREASING = auto()
    INCREASING = auto()
    STAGNANT = auto()

class CombineMethod(StrEnum):
    FISHER = auto()
    STOUFFER = auto()

TestMethod = Literal['page l', 'slope t', 'wilcoxon', 'sign']

@dataclass
class AggregateTrendTestResult:
    """Result of a monotone decreasing trend test."""
    statistic: float
    p_value: float
    significant: bool
    method: str
    trend_type: TrendType

    def __repr__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return f"{self.method} testing for {self.trend_type} trend: statistic={self.statistic:.4f}, p={self.p_value:.6f} ({sig})"

@dataclass
class PerSeriesTrendTestResult:
    """Result of running a per-series trend test and combining the p-values."""

    statistic: float
    p_value: float
    significant: bool
    method: str
    trend_type: TrendType
    combine_method: CombineMethod
    n_series: int
    n_valid: int
    fraction_significant: float
    per_series: pd.DataFrame = field(repr=False)
    """Per-series results indexed by time-series name, with ``p_value`` and
    ``statistic`` columns. Series that could not be tested hold NaN."""

    def __repr__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.method} ({self.combine_method}-combined) testing for {self.trend_type} trend: "
            f"statistic={self.statistic:.4f}, p={self.p_value:.6f} ({sig}); "
            f"{self.fraction_significant:.1%} of {self.n_valid}/{self.n_series} series individually significant"
        )

class TestResultsLatex:
    def __init__(self, *test_results: AggregateTrendTestResult) -> None:
        self.test_results = test_results

    def to_table(self, title: str, star: bool, label: str) -> None:
        print(self._to_table(title, star, label))

    def _to_table(self, title: str, star: bool, label: str) -> str:
        headers = ["Method", "Statistic", "p-value", "Significant"]
        num_cols = len(headers)

        rows = []
        for r in self.test_results:
            sig_str = "Yes" if r.significant else "No"
            row = f"{r.method} & {r.statistic:.4f} & {r.p_value:.6f} & {sig_str} \\\\"
            rows.append(row)

        header_line = " & ".join(headers) + " \\\\"
        rows_block = "\n".join(rows)

        if star:
            col_spec = "@{\\extracolsep\\fill}l" + "c" * (num_cols - 1)
            return (
                "\\begin{table*}[h]\n"
                f"\\caption{{{title}}}\\label{{tab:{label}}}\n"
                f"\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}\n"
                "\\toprule\n"
                f"{header_line}\n"
                "\\midrule\n"
                f"{rows_block}\n"
                "\\botrule\n"
                "\\end{tabular*}\n"
                "\\end{table*}"
            )
        else:
            col_spec = "@{}l" + "l" * (num_cols - 1) + "@{}"
            return (
                "\\begin{table}[h]\n"
                f"\\caption{{{title}}}\\label{{tab:{label}}}\n"
                f"\\begin{{tabular}}{{{col_spec}}}\n"
                "\\toprule\n"
                f"{header_line}\n"
                "\\midrule\n"
                f"{rows_block}\n"
                "\\botrule\n"
                "\\end{tabular}\n"
                "\\end{table}"
            )

def test_stats_matrix(stats_matrix: StatsMatrix, value: str, trend_type: TrendType, test_fun: Callable, **kwargs) -> AggregateTrendTestResult:
    array = _stats_matrix_to_array(stats_matrix, value)
    return test_fun(array, trend_type=trend_type, **kwargs)


def _stats_matrix_to_array(stats_matrix: StatsMatrix, value: str):
    return np.vectorize(lambda s: s[value].item())(stats_matrix)

def slope_ttest(
    data: np.ndarray,
    alpha: float = 0.05,
    trend_type: TrendType = TrendType.DECREASING,
    epsilon: float | None = None,
) -> AggregateTrendTestResult:
    """
    Parametric test for a monotone trend (or stagnancy) across columns.

    Fits a linear regression per row (independent sequence), collecting one
    slope per sequence. Then:

    - For DECREASING / INCREASING: a one-sided one-sample t-test on the slopes
      checks whether the mean slope is significantly negative / positive.
    - For STAGNANT: a Two One-Sided Tests (TOST) equivalence test checks whether
      the mean slope lies inside ``[-epsilon, +epsilon]``, i.e. is practically
      zero. Rejecting the TOST null lets you make a *positive* claim of
      stagnancy (a large p-value from a plain trend test would only mean
      "no trend detected", which is not the same thing).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_sequences, n_positions). Each row is an
        independent sequence; columns are ordered positions.
    alpha : float
        Significance level (default 0.05).
    trend_type : TrendType
        DECREASING, INCREASING, or STAGNANT.
    epsilon : float, optional
        Equivalence margin for the STAGNANT test, expressed as an absolute
        slope (change in value per unit position). A slope whose magnitude is
        below ``epsilon`` is considered negligible. If ``None`` (default), it
        defaults to ``0.1 * std(slopes)`` so the test has a scale-relative
        margin; supply an explicit value when you have a domain-meaningful
        "flat enough" threshold. Ignored for INCREASING / DECREASING.

    Returns
    -------
    TrendTestResult
        For a monotone trend: t-statistic, one-sided p-value, significance flag.
        For STAGNANT: the (larger, i.e. binding) TOST t-statistic, the TOST
        p-value, and whether equivalence to zero slope was established.
    """
    positions = np.arange(data.shape[1])
    slopes = np.array([stats.linregress(positions, row).slope for row in data])

    if trend_type == TrendType.STAGNANT:
        if epsilon is None:
            epsilon = 0.1 * float(np.std(slopes, ddof=1))
        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive for a STAGNANT (equivalence) test; "
                "the slopes have (near) zero variance so no scale-relative "
                "default could be derived. Pass an explicit epsilon."
            )
        # TOST: H0 is |mean slope| >= epsilon (a real trend exists).
        #   lower test  H0: mean <= -epsilon   vs  H1: mean > -epsilon
        #   upper test  H0: mean >=  epsilon   vs  H1: mean <  epsilon
        # Equivalence (stagnancy) is concluded only if BOTH one-sided tests
        # reject, so the overall p-value is the larger of the two.
        t_lower, p_lower = stats.ttest_1samp(slopes, -epsilon, alternative="greater")
        t_upper, p_upper = stats.ttest_1samp(slopes, epsilon, alternative="less")
        if p_lower >= p_upper:
            t_stat, p_value = t_lower, p_lower
        else:
            t_stat, p_value = t_upper, p_upper

        return AggregateTrendTestResult(
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < alpha,
            method="TOST equivalence on slopes",
            trend_type=trend_type,
        )

    t_stat, p_two_sided = stats.ttest_1samp(slopes, 0)
    # One-sided test: H1 is mean slope < 0 (decreasing)
    condition_met = t_stat < 0 if trend_type == TrendType.DECREASING else t_stat > 0
    p_value = p_two_sided / 2 if condition_met else 1 - p_two_sided / 2

    return AggregateTrendTestResult(
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        method="t-test on slopes",
        trend_type=trend_type
    )


def page_trend_test(data: np.ndarray, alpha: float = 0.05, trend_type: TrendType = TrendType.DECREASING) -> AggregateTrendTestResult:
    """
    Non-parametric test for a monotone trend across columns.

    Uses Page's L test, which tests the alternative hypothesis that
    the observations follow a specific monotonic ordering

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_sequences, n_positions). Each row is an
        independent sequence; columns are ordered positions.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    TrendTestResult
        Contains L statistic, p-value, and significance flag.
    """
    k = data.shape[1]
    predicted_ranks = list(range(k, 0, -1) if trend_type == TrendType.DECREASING else range(1, k + 1, 1))

    result = stats.page_trend_test(data, predicted_ranks=predicted_ranks, method="asymptotic")

    return AggregateTrendTestResult(
        statistic=result.statistic,
        p_value=result.pvalue,
        significant=result.pvalue < alpha,
        method="Page's L trend test",
        trend_type=trend_type
    )


def _paired_differences(data: np.ndarray) -> np.ndarray:
    """Turn a two-column matrix into a vector of paired differences.

    For ``SingleStepHierarchies`` the array is ``(n_series, 2)`` where column 0
    is the minimal (base) hierarchy error and column 1 is the one-extra-level
    hierarchy error. The difference ``d = col0 - col1`` is positive when adding
    a level reduces the error.

    DECREASING (error falls col0 -> col1) therefore corresponds to ``d > 0``.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Expected a two-column (n_series, 2) matrix, got shape {data.shape}."
        )
    d = data[:, 0] - data[:, 1]
    return d[np.isfinite(d)]


def _stagnant_margin(diffs: np.ndarray, epsilon: float | None) -> float:
    """Resolve the equivalence margin for a STAGNANT (TOST) test."""
    if epsilon is None:
        epsilon = 0.1 * float(np.std(diffs, ddof=1))
    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive for a STAGNANT (equivalence) test; the "
            "differences have (near) zero variance so no scale-relative default "
            "could be derived. Pass an explicit epsilon."
        )
    return epsilon


def wilcoxon_test(
    data: np.ndarray,
    alpha: float = 0.05,
    trend_type: TrendType = TrendType.DECREASING,
    epsilon: float | None = None,
) -> AggregateTrendTestResult:
    """
    Non-parametric paired test on a two-column matrix (Wilcoxon signed-rank).

    Works on the paired differences ``d = col0 - col1`` (minimal minus
    one-extra-level error). Uses the ranks of the signed differences, so it
    assumes the differences are symmetric about their median but does not
    assume normality.

    - DECREASING: H1 is ``median(d) > 0`` (adding a level reduces error).
    - INCREASING: H1 is ``median(d) < 0``.
    - STAGNANT: TOST equivalence -- two one-sided signed-rank tests against the
      band ``[-epsilon, +epsilon]``, concluding the location of ``d`` is
      practically zero only if both reject. ``epsilon`` (absolute error-metric
      units) defaults to ``0.1 * std(d)`` when not given.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_series, 2).
    alpha : float
        Significance level (default 0.05).
    trend_type : TrendType
        DECREASING, INCREASING, or STAGNANT.
    epsilon : float, optional
        Equivalence margin, only used for STAGNANT.

    Returns
    -------
    TrendTestResult
    """
    d = _paired_differences(data)

    if trend_type == TrendType.STAGNANT:
        eps = _stagnant_margin(d, epsilon)
        # lower: H1 median > -eps   upper: H1 median < +eps
        w_lower, p_lower = stats.wilcoxon(d + eps, alternative="greater")
        w_upper, p_upper = stats.wilcoxon(d - eps, alternative="less")
        statistic, p_value = (w_lower, p_lower) if p_lower >= p_upper else (w_upper, p_upper)
        return AggregateTrendTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < alpha,
            method="Wilcoxon signed-rank TOST equivalence",
            trend_type=trend_type,
        )

    alternative = "greater" if trend_type == TrendType.DECREASING else "less"
    result = stats.wilcoxon(d, alternative=alternative)

    return AggregateTrendTestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=result.pvalue < alpha,
        method="Wilcoxon signed-rank test",
        trend_type=trend_type,
    )


def sign_test(
    data: np.ndarray,
    alpha: float = 0.05,
    trend_type: TrendType = TrendType.DECREASING,
    epsilon: float | None = None,
) -> AggregateTrendTestResult:
    """
    Most assumption-free paired test on a two-column matrix (sign test).

    Works on the paired differences ``d = col0 - col1``, using only their sign
    against a Binomial(n, 0.5) null. Makes no symmetry or distributional
    assumption beyond independence -- robust to skew, at the cost of power
    (magnitude information is discarded).

    - DECREASING: H1 is ``P(d > 0) > 0.5`` (adding a level tends to reduce error).
    - INCREASING: H1 is ``P(d > 0) < 0.5``.
    - STAGNANT: TOST equivalence -- two one-sided sign tests against the band
      ``[-epsilon, +epsilon]``. The lower test asks whether few differences fall
      below ``-epsilon``; the upper whether few exceed ``+epsilon``. Equivalence
      is concluded only if both reject. ``epsilon`` defaults to ``0.1 * std(d)``.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_series, 2).
    alpha : float
        Significance level (default 0.05).
    trend_type : TrendType
        DECREASING, INCREASING, or STAGNANT.
    epsilon : float, optional
        Equivalence margin, only used for STAGNANT.

    Returns
    -------
    TrendTestResult
        The statistic is the number of positive differences used in the
        binomial test (for STAGNANT, the binding side's count).
    """
    d = _paired_differences(data)

    if trend_type == TrendType.STAGNANT:
        eps = _stagnant_margin(d, epsilon)
        # H1 (equivalence): the location of d sits inside (-eps, +eps).
        # lower one-sided: few differences below -eps  -> P(d < -eps) < 0.5
        n_below = int(np.sum(d < -eps))
        n_lower = int(np.sum(d != -eps))
        res_lower = stats.binomtest(n_below, n_lower, 0.5, alternative="less")
        # upper one-sided: few differences above +eps -> P(d > +eps) < 0.5
        n_above = int(np.sum(d > eps))
        n_upper = int(np.sum(d != eps))
        res_upper = stats.binomtest(n_above, n_upper, 0.5, alternative="less")

        if res_lower.pvalue >= res_upper.pvalue:
            statistic, p_value = n_below, res_lower.pvalue
        else:
            statistic, p_value = n_above, res_upper.pvalue

        return AggregateTrendTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < alpha,
            method="Sign test TOST equivalence",
            trend_type=trend_type,
        )

    n_pos = int(np.sum(d > 0))
    n_nonzero = int(np.sum(d != 0))
    # DECREASING -> expect a majority of positive differences.
    alternative = "greater" if trend_type == TrendType.DECREASING else "less"
    result = stats.binomtest(n_pos, n_nonzero, 0.5, alternative=alternative)

    return AggregateTrendTestResult(
        statistic=float(n_pos),
        p_value=float(result.pvalue),
        significant=result.pvalue < alpha,
        method="Sign test",
        trend_type=trend_type,
    )


def _error_tensor(sampled_hierarchies: SampledHierarchies, em: ErrorMetrics,
                  forecast_type: str, granularity: str, error_metric: str,
                  method: str) -> tuple[np.ndarray, list[str]]:
    """Build a ``(n_sequences, n_heights, n_series)`` tensor of raw per-series errors.

    For every sampled-hierarchy cell ``(group, height)`` we select the fully
    specified metrics row (one hierarchy / forecast_type / granularity /
    error_metric / method) and read its raw per-series values, instead of
    collapsing them into an aggregate. The sequence axis (groups) and the
    height axis are preserved so each series can later be tested on its own
    ``(n_sequences, n_heights)`` matrix.

    Returns the tensor and the list of series column names.
    """
    index_matrix = sampled_hierarchies.index_matrix(
        forecast_type=forecast_type,
        granularity=granularity,
        error_metric=error_metric,
        method=method,
    )
    n_groups, n_heights = index_matrix.shape
    ts_columns = em.ts_columns
    n_series = len(ts_columns)

    tensor = np.empty((n_groups, n_heights, n_series), dtype=float)
    for g in range(n_groups):
        for h in range(n_heights):
            # A fully specified index selects exactly one row -> shape (1, n_series).
            values = index_matrix[g, h].values(em)
            tensor[g, h, :] = np.asarray(values, dtype=float).reshape(-1)[:n_series]
    return tensor, ts_columns


def _combine_p_values(p_values: np.ndarray, combine_method: CombineMethod) -> tuple[float, float]:
    """Combine independent one-sided p-values into a single test.

    Returns ``(statistic, combined_p_value)``.

    - Fisher: ``-2 * sum(ln p)`` ~ chi-square with ``2k`` dof. Sensitive to a
      few strongly significant series.
    - Stouffer: mean of the z-scores scaled by ``sqrt(k)``. Sensitive to a
      consistent weak effect across many series.
    """
    # Guard against exact 0/1 which blow up the log / inverse-normal transforms.
    eps = np.finfo(float).eps
    p = np.clip(p_values, eps, 1.0 - eps)

    statistic, combined_p = stats.combine_pvalues(p, method=combine_method)

    return float(statistic), float(combined_p)


def per_series_test(
    test_fun,
    sampled_hierarchies: SampledHierarchies,
    em: ErrorMetrics,
    *,
    forecast_type: str = RECONCILED_FORECASTS,
    granularity: str | Aggregation,
    error_metric: str,
    method: str = 'shr',
    trend_type: TrendType = TrendType.DECREASING,
    combine_method: CombineMethod = CombineMethod.STOUFFER,
    alpha: float = 0.05,
    **kwargs
) -> PerSeriesTrendTestResult:
    """Per-series Page's L trend test across sampled hierarchies, with combined p-value.

    This is the raw-value counterpart of ``l_test_stats``. Instead of collapsing
    all series into one aggregate (e.g. the winsorized mean) *before* running a
    single Page's L test, this:

    1. Builds a ``(n_sequences, n_heights, n_series)`` tensor of the raw
       per-series errors for the selected slice.
    2. Runs Page's L on each series' own ``(n_sequences, n_heights)`` matrix,
       using the sampled hierarchies as independent replicate sequences.
    3. Combines the per-series p-values with Fisher or Stouffer, and also
       reports the fraction of individually significant series.

    Parameters
    ----------
    sampled_hierarchies : SampledHierarchies
        Provides the ``(n_groups, n_heights)`` grid of hierarchies.
    em : ErrorMetrics
        Holds the raw per-series error metrics.
    forecast_type, granularity, error_metric, method : str
        Fully specify which metrics row to read per hierarchy cell.
    trend_type : TrendType
        DECREASING tests that error falls as the hierarchy grows.
    combine_method : CombineMethod
        How to combine the per-series p-values into one.
    alpha : float
        Significance level for both the per-series flags and the combined test.

    Returns
    -------
    PerSeriesTrendTestResult
    """
    tensor, ts_columns = _error_tensor(
        sampled_hierarchies, em,
        forecast_type=forecast_type, granularity=granularity,
        error_metric=error_metric, method=method,
    )
    n_series = tensor.shape[2]

    p_values = np.full(n_series, np.nan)
    statistics = np.full(n_series, np.nan)
    for s in range(n_series):
        matrix = tensor[:, :, s]  # (n_sequences, n_heights)
        # Skip series with any non-finite cell: Page's L cannot rank NaNs.
        if not np.all(np.isfinite(matrix)):
            continue
        # Page's L needs at least 2 positions and 2 replicate sequences.
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            continue
        res = test_fun(matrix, alpha=alpha, trend_type=trend_type, **kwargs)
        p_values[s] = res.p_value
        statistics[s] = res.statistic

    valid = np.isfinite(p_values)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No series yielded a valid per-series trend test ")

    valid_p = p_values[valid]
    fraction_significant = float(np.mean(valid_p < alpha))
    statistic, combined_p = _combine_p_values(valid_p, combine_method)

    per_series = pd.DataFrame(
        {'p_value': p_values, 'statistic': statistics},
        index=pd.Index(ts_columns, name='series'),
    )

    return PerSeriesTrendTestResult(
        statistic=statistic,
        p_value=combined_p,
        significant=combined_p < alpha,
        method="Per-series Page's L trend test",
        trend_type=trend_type,
        combine_method=combine_method,
        n_series=n_series,
        n_valid=n_valid,
        fraction_significant=fraction_significant,
        per_series=per_series,
    )


@dataclass
class PerSeriesTrendTestResultStoreGranularity:
    decreasing: dict[TestMethod, PerSeriesTrendTestResult] = field(default_factory=dict)
    increasing: dict[TestMethod, PerSeriesTrendTestResult] = field(default_factory=dict)
    stagnant: dict[TestMethod, PerSeriesTrendTestResult] = field(default_factory=dict)

@dataclass
class PerSeriesTrendTestResultStore:
    top: PerSeriesTrendTestResultStoreGranularity
    bottom: PerSeriesTrendTestResultStoreGranularity


def run_comprehensive_per_series_tests(sh: SampledHierarchies, em: ErrorMetrics, error_metric: str):
    if not isinstance(sh, SingleStepHierarchies):
        return PerSeriesTrendTestResultStore(
            top=PerSeriesTrendTestResultStoreGranularity(
                    increasing={
                        "page l":  per_series_test(page_trend_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING),
                        "slope t": per_series_test(slope_ttest, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING)
                    },
                    decreasing={
                        "page l":  per_series_test(page_trend_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING),
                        "slope t": per_series_test(slope_ttest, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING)
                    },
                    stagnant={
                        "slope t": per_series_test(slope_ttest, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.STAGNANT)
                    }
                ),
            bottom=PerSeriesTrendTestResultStoreGranularity(
                    increasing={
                        "page l":  per_series_test(page_trend_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING),
                        "slope t": per_series_test(slope_ttest, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING)
                    },
                    decreasing={
                        "page l":  per_series_test(page_trend_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING),
                        "slope t": per_series_test(slope_ttest, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING)
                    },
                    stagnant={
                        "slope t": per_series_test(slope_ttest, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.STAGNANT)
                    }
                )
        )
    else:
        return PerSeriesTrendTestResultStore(
            top=PerSeriesTrendTestResultStoreGranularity(
                    increasing={
                        "wilcoxon":  per_series_test(wilcoxon_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING),
                        "sign": per_series_test(sign_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING)
                    },
                    decreasing={
                        "wilcoxon":  per_series_test(wilcoxon_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING),
                        "sign": per_series_test(sign_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING)
                    },
                    stagnant={
                        "wilcoxon": per_series_test(wilcoxon_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.STAGNANT),
                        "sign": per_series_test(sign_test, sh, em, granularity=TOP_LEVEL, error_metric='nrmse', trend_type=TrendType.STAGNANT)
                    }
                ),
            bottom=PerSeriesTrendTestResultStoreGranularity(
                    increasing={
                        "wilcoxon":  per_series_test(wilcoxon_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING),
                        "sign": per_series_test(sign_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.INCREASING)
                    },
                    decreasing={
                        "wilcoxon":  per_series_test(wilcoxon_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING),
                        "sign": per_series_test(sign_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.DECREASING)
                    },
                    stagnant={
                        "wilcoxon": per_series_test(wilcoxon_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.STAGNANT),
                        "sign": per_series_test(sign_test, sh, em, granularity=BOTTOM_LEVEL, error_metric='nrmse', trend_type=TrendType.STAGNANT)
                    }
                ),
        )