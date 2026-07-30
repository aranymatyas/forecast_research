from dataclasses import dataclass
from typing import Literal
from python.sampled_hierarchies import SampledHierarchies, StatsMatrix

import numpy as np
from scipy import stats
from enum import StrEnum, auto

class TrendType(StrEnum):
    DECREASING = auto()
    INCREASING = auto()

@dataclass
class TrendTestResult:
    """Result of a monotone decreasing trend test."""
    statistic: float
    p_value: float
    significant: bool
    method: str
    trend_type: TrendType

    def __repr__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return f"{self.method} testing for {self.trend_type} trend: statistic={self.statistic:.4f}, p={self.p_value:.6f} ({sig})"

class TestResultsLatex:
    def __init__(self, *test_results: TrendTestResult) -> None:
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

def l_test_stats(stats_matrix: StatsMatrix, value: str, trend_type: TrendType, **kwargs) -> TrendTestResult:
    array = _stats_matrix_to_array(stats_matrix, value)
    return page_trend_test(array, trend_type=trend_type, **kwargs)

def t_test_stats(stats_matrix: StatsMatrix, value: str, trend_type: TrendType, **kwargs) -> TrendTestResult:
    array = _stats_matrix_to_array(stats_matrix, value)
    return slope_ttest(array, trend_type=trend_type, **kwargs)

def _stats_matrix_to_array(stats_matrix: StatsMatrix, value: str):
    return np.vectorize(lambda s: s[value].item())(stats_matrix)

def slope_ttest(data: np.ndarray, alpha: float = 0.05, trend_type: TrendType = TrendType.DECREASING) -> TrendTestResult:
    """
    Parametric test for a monotone trend across columns.

    Fits a linear regression per row (independent sequence), then performs
    a one-sample t-test on the slopes to check if the mean slope is
    significantly negative/positive.

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
        Contains t-statistic, one-sided p-value, and significance flag.
    """
    positions = np.arange(data.shape[1])
    slopes = np.array([stats.linregress(positions, row).slope for row in data])

    t_stat, p_two_sided = stats.ttest_1samp(slopes, 0)
    # One-sided test: H1 is mean slope < 0 (decreasing)
    condition_met = t_stat < 0 if trend_type == TrendType.DECREASING else t_stat > 0
    p_value = p_two_sided / 2 if condition_met else 1 - p_two_sided / 2

    return TrendTestResult(
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        method="t-test on slopes",
        trend_type=trend_type
    )


def page_trend_test(data: np.ndarray, alpha: float = 0.05, trend_type: TrendType = TrendType.DECREASING) -> TrendTestResult:
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
    predicted_ranks = list(range(k, 0, -1) if trend_type == TrendType.DECREASING else range(0, k, 1))

    result = stats.page_trend_test(data, predicted_ranks=predicted_ranks, method="asymptotic")

    return TrendTestResult(
        statistic=result.statistic,
        p_value=result.pvalue,
        significant=result.pvalue < alpha,
        method="Page's L trend test",
        trend_type=trend_type
    )
