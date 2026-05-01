## Proof of theorem

The theorem which will be proved states the following: **Adding Intermediate Aggregation Levels Reduces Forecast Error RMSE Under MinT Reconciliation**

### Notation and Setup

Consider a base period of $n = 28$ days. We observe a scalar time series at daily granularity and wish to forecast at the **top level** (the 28-day aggregate) and the **bottom level** (daily).

A **temporal hierarchy** $\mathcal{H}$ is defined by a set of aggregation levels. Each level $\ell$ specifies how the 28 bottom-level (daily) values are grouped. The relationship between all levels and the bottom level is encoded in a **summing matrix** $S \in \mathbb{R}^{m \times n}$, where $m$ is the total number of time series across all levels (including the bottom level, so $m \geq n$), and $n = 28$. The matrix satisfies:

$$\mathbf{y} = S \mathbf{b}$$

where $\mathbf{b} \in \mathbb{R}^n$ is the vector of true bottom-level values and $\mathbf{y} \in \mathbb{R}^m$ is the full stacked vector of true values at all levels. The bottom $n$ rows of $S$ form the identity $I_n$, and each upper row is a binary (0/1) vector indicating which daily values are summed to form that aggregate.

We produce **base forecasts** $\hat{\mathbf{y}} \in \mathbb{R}^m$ independently at each level. The base forecast errors are:

$$\hat{\mathbf{e}} = \hat{\mathbf{y}} - \mathbf{y}$$

The base forecasts may be **biased**. We write:

$$\mathbb{E}[\hat{\mathbf{e}}] = \boldsymbol{\delta}$$

where $\boldsymbol{\delta} \in \mathbb{R}^m$ is an arbitrary (possibly non-zero) bias vector.

Let $W = \text{Var}(\hat{\mathbf{e}}) \in \mathbb{R}^{m \times m}$ be the covariance matrix of the base forecast errors. Every covariance matrix is positive semi-definite by construction, and in practice it will be positive definite, since for any **constant** $\mathbf{x} \in \mathbb{R}^m$ we write the following:

$$\mathbf{x}' W \mathbf{x} = \text{Var}(\mathbf{x}'(\hat{\mathbf{e}} - \mathbb{E}[\hat{\mathbf{e}}])) = \text{Var}(\mathbf{x}'(\hat{\mathbf{e}} - \boldsymbol{\delta}))$$

To show pisitive semi definity, we just expand the variance and see that a square is always non negative:

$$\text{Var}(\mathbf{x}'(\hat{\mathbf{e}} - \mathbb{E}[\hat{\mathbf{e}}])) = \mathbb{E}\left[(\mathbf{x}'(\hat{\mathbf{e}} - \mathbb{E}[\hat{\mathbf{e}}]))^2\right] \geq 0$$

For $W$ to not be poisitive definite, we would need to show such $\mathbf{x}$ that would make the dot product $\mathbf{x}'(\hat{\mathbf{e}} - \boldsymbol{\delta})$ constant to collapse the variance to zero. The existance of such $\mathbf{x}$ would mean that the $m$ different residuals could be written as a linear combination of other residuals (plus a constant). Because base forecasts are made independently, this is highly unlikely.

---

### MinT Reconciliation

The MinT (Minimum Trace) reconciled forecast is written and defined as:

$$\tilde{\mathbf{y}} = S \underbrace{(S' W^{-1} S)^{-1} S' W^{-1}}_{G} \hat{\mathbf{y}} = S G \hat{\mathbf{y}}$$

where $\hat{\mathbf{y}}$ are the independently made, incoherent forecasts, and $\tilde{\mathbf{y}}$ are the coherent forecasts, and $W = \text{Var}(\hat{\mathbf{y}})$.

This is a **generalised least squares (GLS)** estimator according to [Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2017). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization, Section 2.2](https://robjhyndman.com/papers/MinT.pdf).

It estimates the leaf/bottom level forecasts by projecting the base forecasts onto the coherent subspace where $\{ S G {\mathbf{y}} = S\mathbf{b} : \mathbf{b} \in \mathbb{R}^n \}$, and then applying the summing matrix.

The reconciled bottom-level forecast is $\tilde{\mathbf{b}} = G \hat{\mathbf{y}}$, and any aggregate level's reconciled forecast is obtained by the appropriate rows of $S \tilde{\mathbf{b}}$.

For simplicity, we can define $\mathbf{P}$ to be

$$\mathbf{P} = \mathbf{S}\mathbf{G}$$

#### 2.1 Unbiasedness of MinT

**Assumption 1 (Unbiasedness).** We assume the base forecasts are unbiased: $\mathbb{E}[\hat{\mathbf{e}}] = \boldsymbol{\delta} = \mathbf{0}$. From here we need to establish that if this assumption is true, then the reconciled forecasts produced from the unbiased base ones, stay unbiased.

From the paper [Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization, Section 2.1](https://robjhyndman.com/papers/MinT.pdf) we know that minT reconciliation is an unbiased projection since

$$\mathbf{S}\mathbf{G}\mathbf{S} = \mathbf{S}$$

meaning for any coherent forecast $y_c$

$$\mathbf{P}\mathbf{y_c} = \mathbf{y_c}$$

We know that the reconciled and base errors can be calculated like:

$$\tilde{\mathbf{e}} = \mathbf{y} - \tilde{\mathbf{y}}$$

$$\hat{\mathbf{e}} = \mathbf{y} - \hat{\mathbf{y}}$$

and if we substitute $\tilde{\mathbf{y}} = \mathbf{P}\hat{\mathbf{y}}$ and the base error formula

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{P}\hat{\mathbf{y}}$$

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{P}(\mathbf{y} - \hat{\mathbf{e}})$$

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{P}\mathbf{y} + \mathbf{P}\hat{\mathbf{e}}$$

Using $\mathbf{P}\mathbf{y} = \mathbf{y}$

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{y} + \mathbf{P}\hat{\mathbf{e}}$$

$$\tilde{\mathbf{e}} = \mathbf{P}\hat{\mathbf{e}}$$

if $\hat{\mathbf{e}}$ is zero, then

$$\tilde{\mathbf{e}} = \mathbf{P}(\mathbf{0}) = \mathbf{0}$$


Therefore $\mathbb{E}[\tilde{\mathbf{y}}] = \mathbf{y}$. MinT **preserves unbiasedness**: if the base forecasts are unbiased, so are the reconciled forecasts.

#### 2.2 Covariance of the Reconciled Forecasts

- The reconciled forecasts are $\tilde{\mathbf{y}} = \mathbf{P}\hat{\mathbf{y}}$.
- The variance of the base forecasts is $\text{Var}(\hat{\mathbf{y}}) = \mathbf{W}$.
- The variance of the reconciled forecasts is $\text{Var}(\tilde{\mathbf{y}}) = \mathbf{V}$.

Then because variance of vectors has the following property:

$$\text{Var}(\mathbf{P}\hat{\mathbf{y}}) = \mathbf{P} \text{Var}(\hat{\mathbf{y}}) \mathbf{P}'$$

From this, we know that the reconciled forecast variance $\mathbf{V}$ is

$$\mathbf{V} = \mathbf{P} \mathbf{W} \mathbf{P}'$$

- We know that: $\mathbf{P} = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1}$
- We also know that $\mathbf{W}$ is symmetrical, therefore $\mathbf{W}=\mathbf{W}'$. The inverse of a symmetrical matrix is symmetrical, because

$$\mathbf{W} \mathbf{W}^{-1} = \mathbf{I}$$

$$(\mathbf{W} \mathbf{W}^{-1})' = \mathbf{I}'$$

$$(\mathbf{W}^{-1})' \mathbf{W}' = \mathbf{I}$$

$$(\mathbf{W}^{-1})' \mathbf{W} = \mathbf{I}$$

$$\mathbf{W}^{-1}=(\mathbf{W}^{-1})'$$

Applying the transpose rule of product matrices where we reverse the order and transpose each term individually:

$$\mathbf{P}' = (\mathbf{W}^{-1})' (\mathbf{S}')' \left((\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\right)' \mathbf{S}' = \mathbf{W}^{-1} \mathbf{S} (\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1} \mathbf{S}'$$

If we substitue $\mathbf{P}$ and $\mathbf{P}'$ back into our variance formula

$$\mathbf{V} = \left[ \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1} \right] \mathbf{W} \left[ \mathbf{W}^{-1}\mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}' \right]$$

Then simplify by eliminating the inverse pair of $\mathbf{W}^{-1}$ and $\mathbf{W}$ in the middle

$$\mathbf{V} = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1} \left[ \mathbf{S}'\mathbf{W}^{-1} \mathbf{S} \right] (\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'$$

Then simplify by eliminating the inverse pair of $(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})$ and $(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}$ It turns out that:

$$\mathbf{V} = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'$$

---

### MSE Decomposition

For any unknown parameter $\theta$ and its estimator $\hat{\theta}$, we know the following definitions

- Mean Squared Error: $\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2]$
- Variance: $\text{Var}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]$
- Bias: $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$

We can add and substract the expected value of the estimator in the formula of MSE

$$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}] + \mathbb{E}[\hat{\theta}] - \theta)^2]$$

$$\text{MSE}(\hat{\theta}) = \mathbb{E}[\left( (\hat{\theta} - \mathbb{E}[\hat{\theta}]) + (\mathbb{E}[\hat{\theta}] - \theta) \right)^2]$$

Then we expand the square

$$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2 + 2(\hat{\theta} - \mathbb{E}[\hat{\theta}])(\mathbb{E}[\hat{\theta}] - \theta) + (\mathbb{E}[\hat{\theta}] - \theta)^2]$$

We use the linearity of the expected value operator

$$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2] + \mathbb{E}[2(\hat{\theta} - \mathbb{E}[\hat{\theta}])(\mathbb{E}[\hat{\theta}] - \theta)] + \mathbb{E}[(\mathbb{E}[\hat{\theta}] - \theta)^2]$$

- The first term is the variance of the estimator: $\text{Var}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]$
- The last term has a constant as operand of the expected value, which turn out to be the bias sqaured: $\mathbb{E}[(\mathbb{E}[\hat{\theta}] - \theta)^2] = (\mathbb{E}[\hat{\theta}] - \theta)^2 = \text{Bias}(\hat{\theta})^2$
- The middle term has a constant part which can be pulled out

$$\mathbb{E}[2(\hat{\theta} - \mathbb{E}[\hat{\theta}])(\mathbb{E}[\hat{\theta}] - \theta)] = 2(\mathbb{E}[\hat{\theta}] - \theta) \cdot \mathbb{E}[\hat{\theta} - \mathbb{E}[\hat{\theta}]]$$

Then we use linearity on the second factor, and see that there is a constant as operand of expected value again

$$\mathbb{E}[\hat{\theta} - \mathbb{E}[\hat{\theta}]] = \mathbb{E}[\hat{\theta}] - \mathbb{E}[\mathbb{E}[\hat{\theta}]] = \mathbb{E}[\hat{\theta}] - \mathbb{E}[\hat{\theta}] = 0$$

Hence the middle term turns out to be zero, leaving the MSE error metric to be

$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2$$

For our case, since we assume that our base forecast are unbaised, and because of that our reconciled forecasts are unbiased, the formula reduces to

$$\text{MSE}({y}) = \text{Var}({y}) + \text{Bias}({y})^2 = \text{Var}({y}) + 0 = \text{Var}({y})$$

From that we can also state that

$$\text{RMSE}(y) = \sqrt{\text{Var}({y})}$$

---

### Positive Semi-Definity of covariance differences

- Let $\hat{\mathbf{e}}$ be the base forecast errors, and $\mathbf{W} = \text{Var}(\hat{\mathbf{e}})$.
- Let $\mathbf{P}$ be the MinT projection matrix: $\mathbf{P} = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1}$.
- Let $\tilde{\mathbf{e}}$ be the reconciled errors, where $\tilde{\mathbf{e}} = \mathbf{P}\hat{\mathbf{e}}$.

With the following algebraic trick

$$\hat{\mathbf{e}} = \mathbf{P}\hat{\mathbf{e}} + (\hat{\mathbf{e}} - \mathbf{P}\hat{\mathbf{e}})$$

We can write

$$\hat{\mathbf{e}} = \mathbf{P}\hat{\mathbf{e}} + (\mathbf{I} - \mathbf{P})\hat{\mathbf{e}}$$

If we calculate the covariance of the two terms on the right hand side

$$\text{Cov} \left( \mathbf{P}\hat{\mathbf{e}}, (\mathbf{I} - \mathbf{P})\hat{\mathbf{e}} \right)$$

$$= \mathbf{P} \text{Var}(\hat{\mathbf{e}}) (\mathbf{I} - \mathbf{P})'$$

$$= \mathbf{P} \mathbf{W} (\mathbf{I} - \mathbf{P}')$$

$$= \mathbf{P}\mathbf{W} - \mathbf{P}\mathbf{W}\mathbf{P}'$$

From section 2.2 we know that

$$\mathbf{P}\mathbf{W}\mathbf{P}' = \mathbf{V}$$

And if we expand the projection matrix $\mathbf{P}$ we can simplify

$$\mathbf{P}\mathbf{W} = \left[ \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1} \right] \mathbf{W}$$

$$\mathbf{P}\mathbf{W} = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'$$

From section 2.2 we know that

$$\mathbf{V} = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}' = \mathbf{P}\mathbf{W}$$

Substituting back our finding into the covariance

$$\text{Cov} \left( \mathbf{P}\hat{\mathbf{e}}, (\mathbf{I} - \mathbf{P})\hat{\mathbf{e}} \right) = \mathbf{V} - \mathbf{V} = \mathbf{0}$$

We see that the two terms are uncorrelated. Because of this, if we then take the covariance of both sides of our original equation, the right side just becomes the sum of the two covariances.

$$\hat{\mathbf{e}} = \mathbf{P}\hat{\mathbf{e}} + (\mathbf{I} - \mathbf{P})\hat{\mathbf{e}}$$

$$\text{Var}(\hat{\mathbf{e}}) = \text{Var}(\mathbf{P}\hat{\mathbf{e}}) + \text{Var}((\mathbf{I} - \mathbf{P})\hat{\mathbf{e}})$$

$$\mathbf{W} = \mathbf{V} + \text{Var}((\mathbf{I} - \mathbf{P})\hat{\mathbf{e}})$$

$$\mathbf{W} - \mathbf{V} = \text{Var}((\mathbf{I} - \mathbf{P})\hat{\mathbf{e}})$$

From section 1. we know that, since on the right side there is a covariance matrix, we know that it positive semi-definite matrix, therefore $\mathbf{W} - \mathbf{V}$ must be PSD as well.

### Core Theorem

#### Monotonicity of Variance

Let $\mathbf{\Delta} = \mathbf{W} - \mathbf{V}$. Because we proved that $\mathbf{\Delta}$ is a PSD matrix, it must satisfy the quadratic form condition for any non-zero vector $\mathbf{x}$:

$$\mathbf{x}' \mathbf{\Delta} \mathbf{x} \ge 0$$

To isolate the variance of any specific node of any aggregation level $i$ (e.g., let $i=1$ denote the top-level aggregate's single node), we define $\mathbf{x}$ as the standard basis vector $\mathbf{e}_i$.

Applying this basis vector to the quadratic form extracts the $i$-th diagonal element of the matrix. Because $\mathbf{\Delta}$ is PSD, the following equation must hold true.

$$\mathbf{e}_i' \mathbf{\Delta} \mathbf{e}_i = \Delta_{ii}$$

$$\Delta_{ii} \ge 0$$

$$(W_{ii} - V_{ii}) \ge 0$$

$$V_{ii} \le W_{ii}$$

Where both $V_{ii}$ and  $W_{ii}$ are scalar values. These values specifically are the $i$-th aggregation node's error variances before and after reconciliation. Hence we know that on node of any aggregation level, reconciliation does not increase the variance of the errors.

#### Introducing more levels

Let $\mathbf{S}_k$ represent the summing matrix of a temporal hierarchy $\mathcal{H}_k$ with $k$ intermediate levels (levels between the leafs and the root). When a new node of an aggregation level is added, the summing matrix expands by adding new rows, becoming $\mathbf{S}_{k+1}$.

The MinT algorithm mathematically formulates forecast reconciliation as a Generalized Least Squares (GLS) regression problem. In this framework, adding intermediate temporal levels introduces new rows to the summing matrix $\mathbf{S}$. Each new row acts as an additional linear constraint - a strict aggregation rule - that the final reconciled forecasts (the solutions to the problem) must satisfy. Because the algorithm must now optimize subject to more restrictions, the set of mathematically allowable forecasts (the feasible parameter space) under the expanded hierarchy $\mathbf{S}_{k+1}$ shrinks (or does not expand at least), becoming a subspace of the parameter space defined by the simpler hierarchy $\mathbf{S}_k$. This means that any reconciled forecasts produced by $k+1$ aggregation nodes, will also satisfy the constraints of the 'base' $k$ aggregations.

For our use case here, we now assume that we add nodes/aggregation levels to the the reconciliation process, but we discard their corresponding reconciled forecast values, so that the number of output forecasts match before and after the introducing of new nodes. This is to make sure that the sizes of the covariance matrices also match.

Following the properties of Restricted Least Squares [Chan, Univ. of Sydney](https://www.maths.usyd.edu.au/u/jchan/GLM/RestrictedLeastSquares.pdf), imposing additional valid linear constraints on an optimal GLS estimator results in a variance matrix equal to the unrestricted variance minus a positive semi-definite matrix. Therefore, if $\mathbf{V}_k$ and $\mathbf{V}_{k+1}$ are the reconciled covariance matrices for $k$ and $k+1$ levels respectively (we call the case with $k$ unrestricted), we can write that

$$\mathbf{V}_{k+1} = \mathbf{V}_k - \text{<A PSD Matrix>}$$

$$\mathbf{V}_k - \mathbf{V}_{k+1} = \text{<A PSD Matrix>}$$

Applying the diagonal extraction to this new PSD difference, we can isolate the variance at any node of an aggregation level, to see that the variance does not increase, by adding more constraints, or aggregation levels.

$$(V_k)_{ii} \ge (V_{k+1})_{ii}$$

#### Conclusion

Since we have shown that the NRMSE errors of individual nodes of aggregation levels (e.g. $week1$ node of weekly aggregation) do not increase by having more of them, we know that the average error of whole aggregation levels do not increase either.

We have shown that, in theory, given that the base forecasts are unbiased, we can guarantee, that by adding any kind of additional aggregation levels, to any hierarchy cannot increase the variance of any forecast, at any level, meaning it cannot increase the RMSE metric of it either. In the research we studied NRMSE, as in normalised RMSE to make the metrics more comparable across different datasets.

The normalisation meant that the RMSE value at a granularity was divided by the variance of the training data at that granularity, which of course is a constant, and independent of what kind of hierarchy was built on said dataset.

Therefore the NRMSE metric cannot increase by adding more hierarchy levels.

### In practice

The theorem explains how the mean of the NRMSE errors decreased at both top level and bottom level hierarchy. We have also seen evidence for increase of the median at the same time, which could be attributed to the fact that the distribution of the NRMSE errors were probably tightened and pulled together, hence possibly increasing the median.

\newpage