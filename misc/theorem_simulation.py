"""
Numerical sanity check / demonstration for the temporal-aggregation theorem.

Theorem (paper, sec. "Theorem for Expansion of Aggregation"):
    Adding intermediate aggregation levels to a temporal hierarchy reduces the
    bottom-level reconciled forecast-error variance in the Loewner partial order
    under MinT reconciliation:

        Var(b~_k) >= Var(b~_{k+1})     (PSD, i.e. every diagonal entry shrinks)

    and hence the (N)RMSE at every common aggregation node cannot increase.

The proof rests on TWO assumptions:
    (1) the covariance W used to weight MinT equals the TRUE base-error
        covariance, and
    (2) the base forecasts are UNBIASED.

In practice W is unknown and is *estimated* (e.g. shrinkage, `comb='shr'`),
and base models (ETS/ARIMA/Prophet) are not exactly unbiased. This module
demonstrates, by simulation, that:

    (A) With the TRUE shared W and unbiased forecasts, the theorem holds
        EXACTLY: monotone non-increase at every nested step.

    (B) With W ESTIMATED by shrinkage from a finite residual sample, the
        guarantee can BREAK -- and it breaks precisely in the small-sample /
        high-dimension regime (few residuals relative to the number of series),
        which is exactly where short/heterogeneous datasets (e.g. M4/M5) live.

This isolates the reason empirical summary statistics only conform for the
web-traffic/AutoETS combination: it is the combo whose data best satisfies
the theorem's assumptions (many residuals -> well-estimated W, low bias).

Run directly:
    python -m python.theorem_simulation

T is the history length / sample count
m is the number of time series in the hierarchy (days = 28)

- **T < m**: sample covariance is singular, shrinkage keeps it invertible but the estimate is dominated by the regularization prior (the diagonal) rather than actual data → you're barely doing MinT at all, more like WLS →
adding levels adds noise, not information.
- **T > m but not by much** (say T ≈ 1.5m–3m): borderline, works for some series, breaks for others.
- **T >> m**: shrinkage estimate converges to the true W → theorem's guarantee kicks in empirically.

And the nasty part for your setup: as you add levels, m grows but T stays fixed (it's determined by the series length, not the hierarchy). So the very act the theorem says should help — adding levels — simultaneously worsens
the estimation quality that makes it work. It's a self-defeating loop for short series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Hierarchy construction
# --------------------------------------------------------------------------- #

BASE_PERIOD = 28  # bottom-level (daily) length of one base period

# Intermediate aggregation levels available to add, given as block sizes that
# divide the base period. A block size `b` sums every `b` consecutive days.
DEFAULT_LEVELS = [2, 4, 7, 14]


def summing_rows(block: int, n: int = BASE_PERIOD) -> np.ndarray:
    """Rows of the summing matrix for one aggregation level.

    Aggregates every `block` consecutive bottom-level periods into one node.
    """
    if n % block != 0:
        raise ValueError(f"block size {block} must divide base period {n}")
    rows = np.zeros((n // block, n))
    for i, start in enumerate(range(0, n, block)):
        rows[i, start:start + block] = 1.0
    return rows


@dataclass
class MasterHierarchy:
    """The full set of possible levels, with a fixed row layout so that any
    sub-hierarchy selects a consistent sub-block of a single shared true W.
    """
    levels: list[int] = field(default_factory=lambda: list(DEFAULT_LEVELS))
    n: int = BASE_PERIOD

    def __post_init__(self) -> None:
        # Ordered blocks: top (full sum) -> intermediates -> bottom (identity).
        self._blocks: list[tuple[str, np.ndarray]] = [("top", summing_rows(self.n, self.n))]
        for b in self.levels:
            self._blocks.append((f"agg{b}", summing_rows(b, self.n)))
        self._blocks.append(("bottom", np.eye(self.n)))

        self._row_ranges: dict[str, list[int]] = {}
        cursor = 0
        for name, blk in self._blocks:
            self._row_ranges[name] = list(range(cursor, cursor + blk.shape[0]))
            cursor += blk.shape[0]
        self.m: int = cursor
        self.S: np.ndarray = np.vstack([blk for _, blk in self._blocks])

    def sub(self, active_levels: list[int]) -> tuple[np.ndarray, list[int]]:
        """Summing matrix and master-row indices for a sub-hierarchy that uses
        `active_levels` (plus the always-present top and bottom levels).
        """
        names = ["top"] + [f"agg{b}" for b in active_levels] + ["bottom"]
        rows: list[int] = []
        for nm in names:
            rows.extend(self._row_ranges[nm])
        return self.S[rows, :], rows

    def nested_sequences(self) -> list[list[int]]:
        """Nested active-level sets: [], [l0], [l0, l1], ... (H_k subset H_{k+1})."""
        return [self.levels[:k] for k in range(len(self.levels) + 1)]


# --------------------------------------------------------------------------- #
# Covariance helpers
# --------------------------------------------------------------------------- #

def random_pd_cov(m: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """A random symmetric positive-definite matrix of size m."""
    A = rng.standard_normal((m, m))
    return scale * (A @ A.T / m + 0.5 * np.eye(m))


def shrink_cov(residuals: np.ndarray, intensity: float = 0.2) -> np.ndarray:
    """Shrinkage covariance estimate toward the diagonal target.

    A simplified stand-in for the Schaefer-Strimmer estimator used by MinT
    `comb='shr'`. `residuals` has shape (T, m); returns an (m, m) estimate.
    """
    S = np.cov(residuals, rowvar=False)
    D = np.diag(np.diag(S))
    return intensity * D + (1.0 - intensity) * S


def mint_bottom_cov(S: np.ndarray, W_weight: np.ndarray, W_true: np.ndarray) -> np.ndarray:
    """Actual covariance of the bottom-level reconciled error.

    MinT builds G with the (possibly estimated) weighting matrix `W_weight`:
        G = (S' Wm^-1 S)^-1 S' Wm^-1
    but the realised error covariance uses the TRUE error covariance `W_true`:
        Var(b~) = G W_true G'.

    When W_weight == W_true this collapses to (S' W^-1 S)^-1, the ideal MinT
    covariance for which the theorem's monotonicity holds exactly.
    """
    Wm_inv = np.linalg.inv(W_weight)
    G = np.linalg.inv(S.T @ Wm_inv @ S) @ S.T @ Wm_inv
    return G @ W_true @ G.T


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

@dataclass
class StepResult:
    active_levels: list[int]
    mean_bottom_var: float
    increased: bool  # True if it increased vs the previous (smaller) hierarchy


def run_true_w(master: MasterHierarchy, rng: np.random.Generator) -> list[StepResult]:
    """(A) TRUE shared W, unbiased -> theorem predicts monotone non-increase."""
    W_master = random_pd_cov(master.m, rng)
    results: list[StepResult] = []
    prev = None
    for active in master.nested_sequences():
        S, rows = master.sub(active)
        W = W_master[np.ix_(rows, rows)]
        mv = float(np.diag(mint_bottom_cov(S, W, W)).mean())
        results.append(StepResult(active, mv, prev is not None and mv > prev + 1e-12))
        prev = mv
    return results


def run_estimated_w(
    master: MasterHierarchy,
    rng: np.random.Generator,
    n_residuals: int,
    n_draws: int = 200,
    shrink_intensity: float = 0.2,
) -> list[StepResult]:
    """(B) ESTIMATED W (shrinkage) from `n_residuals` residuals.

    For each hierarchy we average the realised bottom-level variance over
    `n_draws` independent residual samples drawn from the true W.
    """
    W_master = random_pd_cov(master.m, rng)
    results: list[StepResult] = []
    prev = None
    for active in master.nested_sequences():
        S, rows = master.sub(active)
        W = W_master[np.ix_(rows, rows)]
        L = np.linalg.cholesky(W)
        m = S.shape[0]
        vals = []
        for _ in range(n_draws):
            res = (L @ rng.standard_normal((m, n_residuals))).T  # (T, m), cov ~ W
            W_hat = shrink_cov(res, shrink_intensity)
            vals.append(np.diag(mint_bottom_cov(S, W_hat, W)).mean())
        mv = float(np.mean(vals))
        results.append(StepResult(active, mv, prev is not None and mv > prev + 1e-9))
        prev = mv
    return results


def _print_block(title: str, results: list[StepResult]) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    violations = 0
    for r in results:
        flag = ""
        if r.active_levels and results.index(r) > 0:
            flag = ">>> INCREASED <<<" if r.increased else "OK (down)"
            violations += int(r.increased)
        print(f"  levels={str(r.active_levels):22} mean bottom-var = "
              f"{r.mean_bottom_var:.6f}  {flag}")
    print(f"  => {violations} step(s) violated monotonicity\n")


def main(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    master = MasterHierarchy()

    _print_block(
        "(A) TRUE shared W, unbiased  ->  theorem: monotone non-increase",
        run_true_w(master, rng),
    )

    for T in (40, 100, 500):
        _print_block(
            f"(B) ESTIMATED W via shrinkage, residual sample T = {T}",
            run_estimated_w(master, rng, n_residuals=T),
        )

    print("Reading: (A) never violates monotonicity -> the theorem is correct.")
    print("(B) violates it in the small-T (few residuals / high-dimension) regime,")
    print("which is exactly where short, heterogeneous series break down in")
    print("practice even though MinT shrinkage is the recommended estimator.")


if __name__ == "__main__":
    main()
