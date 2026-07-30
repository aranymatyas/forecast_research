# Simulation Plan: Theorem Validation via Synthetic Data

## Goal

Demonstrate that the temporal-aggregation theorem holds exactly when its assumptions are met (true W, unbiased forecasts), and characterize precisely when and why it breaks in practice (estimated W from limited residuals).

## Key Insight

The experimenter can generate thousands of replications of the same DGP, accumulating enough forecast error vectors to build an oracle "true W" (population covariance). MinT, inside the pipeline, only ever sees one series' worth of in-sample residuals (~T observations) and must estimate the same m×m covariance via shrinkage. This asymmetry isolates the effect of covariance estimation quality on the theorem's practical validity.

## Setup

### Data Generating Process (DGP)

- Pick a known stochastic process at daily granularity (e.g., ETS(A,A,A) with weekly seasonality, or ARIMA(1,1,1)).
- Fix the parameters (smoothing constants, innovation variance σ², initial states).
- The DGP is NOT the forecasting model — they can match (unbiased case) or differ (biased case).

### Controlled Variables

| Variable | Controls | Values to test |
|----------|----------|----------------|
| Series length | T (residuals available for estimation) | 50, 100, 200, 500, 1000 |
| Number of intermediate levels | m (dimension of W) | 1, 2, 3, 4, 5+ levels |
| DGP ↔ model match | Bias | Match (ETS→ETS) vs mismatch (ARIMA→ETS) |
| Signal-to-noise ratio | Residual structure quality | Low/medium/high σ² |

### Output

A 2D landscape (T × m) showing where the theorem holds empirically and where it breaks, interpretable as a T/m ratio threshold.

## Procedure

### Step 1: Estimate the Oracle "True W"

1. Fix a DGP with known parameters.
2. Generate R = 5000–10000 replications of the same series length.
3. For each replication:
   - Aggregate to all hierarchy levels.
   - Fit the base model at each level.
   - Produce the forecast for one base period (28 days).
   - Record the forecast error vector ê_r ∈ ℝ^m (one error per hierarchy node).
4. Stack into a matrix of shape (R, m).
5. Sample covariance of that matrix → **oracle W** (negligible estimation error with R this large).

### Step 2: Run Individual Replications Through the Pipeline

For each single replication (using the actual framework's data structures):

- **(A) Oracle reconciliation**: Feed the oracle W directly into MinT → theorem should hold exactly (monotone non-increase at every nested step).
- **(B) Shrinkage reconciliation**: Use `comb='shr'` from that single series' ~T residuals → may break, depending on T/m ratio.

Same data, same pipeline, different W.

### Step 3: Vary Series Length

Repeat Steps 1–2 for different series lengths (T = 50, 100, 200, 500, 1000). For each length:

- Compute conformism rate (% of nested steps where NRMSE decreases).
- Compare oracle vs shrinkage.
- Identify the T/m threshold where shrinkage starts reliably conforming.

### Step 4: Vary Hierarchy Complexity

Repeat with different numbers of intermediate levels added, keeping series length fixed. This shows how growing m (while T stays constant) degrades performance.

### Step 5 (Optional): Bias Experiment

- **Unbiased**: DGP matches model (generate from ETS, fit ETS).
- **Biased**: DGP mismatches model (generate from ARIMA, fit ETS).

Show that even with oracle W, bias can cause MSE to increase (because MSE = Var + Bias², and the theorem only guarantees Var reduction).

## Expected Results

| Condition | Oracle W | Shrinkage Ŵ |
|-----------|----------|-------------|
| T >> m, unbiased | ✓ monotone | ✓ monotone |
| T >> m, biased | Var ↓ but MSE may ↑ | Var ↓ but MSE may ↑ |
| T ≈ m, unbiased | ✓ monotone | Borderline |
| T < m, unbiased | ✓ monotone | ✗ breaks |
| T < m, biased | Var ↓ but MSE may ↑ | ✗ breaks badly |

## Practitioner Takeaway

"Before expanding a temporal hierarchy, verify that the available residual sample size T exceeds the resulting number of hierarchy nodes m by a sufficient margin (empirically: T/m > X). Otherwise, the covariance estimation noise overwhelms the theoretical variance reduction."

## Paper Structure (Revised)

1. **Theorem** — mathematical proof (as is).
2. **Simulation: assumption sensitivity** — this plan. Shows theorem holds with oracle W, characterizes breakdown under estimation. Produces the T/m threshold.
3. **Real data experiments** — web-traffic/AutoETS conforms (long series ≈ high T/m, low bias), other combos fail (short series or high bias). Framed as empirical confirmation of the simulation's predictions, not as "theorem failure."

## Implementation Notes

- Generate synthetic series in the exact format the existing framework expects (daily time series → TemporalHierarchy → base forecasting → reconciliation → error metrics).
- The only non-standard step: injecting the oracle W into the reconciliation instead of letting it estimate from residuals.
- Check whether `foreco.csrec` accepts a custom covariance matrix, or if a wrapper is needed.
