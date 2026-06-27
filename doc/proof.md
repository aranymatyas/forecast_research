## Proof of theorem

The theorem which will be proved states the following: **Adding Intermediate Aggregation Levels Reduces Forecast Error RMSE, at All Aggregation Levels, Under MinT Reconciliation**

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

Let $\text{Var}(\hat{\mathbf{e}}) \in \mathbb{R}^{m \times m}$ be the covariance matrix of the base forecast errors. Every covariance matrix is positive semi-definite by construction, since for any **constant** $\mathbf{x} \in \mathbb{R}^m$ we write the following:

$$\mathbf{x}' \text{Var}(\hat{\mathbf{e}}) \mathbf{x} = \text{Var}(\mathbf{x}'(\hat{\mathbf{e}} - \mathbb{E}[\hat{\mathbf{e}}])) = \text{Var}(\mathbf{x}'(\hat{\mathbf{e}} - \boldsymbol{\delta}))$$

To show positive semi definity, we just expand the variance and see that a square is always non-negative:

$$\text{Var}(\mathbf{x}'(\hat{\mathbf{e}} - \mathbb{E}[\hat{\mathbf{e}}])) = \mathbb{E}\left[(\mathbf{x}'(\hat{\mathbf{e}} - \mathbb{E}[\hat{\mathbf{e}}]))^2\right] \geq 0$$

For $\text{Var}(\hat{\mathbf{e}})$ to not be positive definite, we would need to show such $\mathbf{x}$ that would make the dot product $\mathbf{x}'(\hat{\mathbf{e}} - \boldsymbol{\delta})$ constant to collapse the variance to zero. The existance of such $\mathbf{x}$ would mean that the $m$ different residuals could be written as a linear combination of other residuals (plus a constant). Because base forecasts are made independently, this is highly unlikely.

---

### MinT Reconciliation

The MinT (Minimum Trace) reconciled forecast is written and defined as:

$$\tilde{\mathbf{y}} = S \underbrace{(S' \text{Var}(\hat{\mathbf{y}})^{-1} S)^{-1} S' \text{Var}(\hat{\mathbf{y}})^{-1}}_{G} \hat{\mathbf{y}} = S G \hat{\mathbf{y}}$$

where $\hat{\mathbf{y}}$ are the independently made, incoherent forecasts, and $\tilde{\mathbf{y}}$ are the coherent forecasts.

This is a **generalised least squares (GLS)** estimator [@hyndmanmint].

It estimates the leaf/bottom level forecasts by projecting the base forecasts onto the coherent subspace where $\{ S G {\mathbf{y}} = S\mathbf{b} : \mathbf{b} \in \mathbb{R}^n \}$, and then applying the summing matrix.

The reconciled bottom-level forecast is $\tilde{\mathbf{b}} = G \hat{\mathbf{y}}$, and any aggregate level's reconciled forecast is obtained by the appropriate rows of $S \tilde{\mathbf{b}}$.

For simplicity, we can define $\mathbf{P}$ to be

$$\mathbf{P} = \mathbf{S}\mathbf{G}$$

#### 2.1 Unbiasedness of MinT

**Assumption 1 (Unbiasedness).** We assume the base forecasts are unbiased: $\mathbb{E}[\hat{\mathbf{e}}] = \boldsymbol{\delta} = \mathbf{0}$. From here we need to establish that if this assumption is true, then the reconciled forecasts produced from the unbiased base ones, stay unbiased.

We know that minT reconciliation is an unbiased projection [@hyndmanmint] since

$$\mathbf{S}\mathbf{G}\mathbf{S} = \mathbf{S}$$

meaning for any coherent forecast $y_c$

$$\mathbf{P}\mathbf{y_c} = \mathbf{y_c}$$

We know that the reconciled and base errors can be calculated like:

$$\tilde{\mathbf{e}} = \mathbf{y} - \tilde{\mathbf{y}}$$

$$\hat{\mathbf{e}} = \mathbf{y} - \hat{\mathbf{y}}$$

and if we substitute $\tilde{\mathbf{y}} = \mathbf{P}\hat{\mathbf{y}}$ in the base error formula

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{P}\hat{\mathbf{y}}$$

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{P}(\mathbf{y} - \hat{\mathbf{e}})$$

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{P}\mathbf{y} + \mathbf{P}\hat{\mathbf{e}}$$

Using $\mathbf{P}\mathbf{y} = \mathbf{y}$

$$\tilde{\mathbf{e}} = \mathbf{y} - \mathbf{y} + \mathbf{P}\hat{\mathbf{e}}$$

$$\tilde{\mathbf{e}} = \mathbf{P}\hat{\mathbf{e}}$$

if $\mathbb{E}[\hat{\mathbf{e}}]$ is zero, then

$$\mathbb{E}[\tilde{\mathbf{e}}] = \mathbb{E}[\mathbf{P}\hat{\mathbf{e}}]$$

$$\mathbb{E}[\tilde{\mathbf{e}}] = \mathbf{P}\mathbb{E}[\hat{\mathbf{e}}]$$

$$\mathbb{E}[\tilde{\mathbf{e}}] = \mathbf{P}(\mathbf{0}) = \mathbf{0}$$

Therefore MinT **preserves unbiasedness**: if the base forecasts are unbiased, so are the reconciled forecasts.

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

$$\text{MSE}({\tilde{\mathbf{y}}}) = \text{Var}({\tilde{\mathbf{y}}}) + \text{Bias}({\tilde{\mathbf{y}}})^2 = \text{Var}({\tilde{\mathbf{y}}}) + 0 = \text{Var}({\tilde{\mathbf{y}}})$$

From that we can also state that

$$\text{RMSE}(\tilde{\mathbf{y}}) = \sqrt{\text{Var}({\tilde{\mathbf{y}}})}$$

---

### Core Theorem

#### Introducing more levels

The MinT algorithm mathematically formulates forecast reconciliation as a Generalized Least Squares (GLS) regression problem. In this framework, the model is written as:

$$\hat{\mathbf{y}}=\mathbf{S}\mathbf{b}+\mathbf{e}$$

where we regress the base forecasts $\hat{\mathbf{y}}$ on the summing matrix $\mathbf{S}$ to estimate the bottom-level coherent forecasts $\mathbf{b}$. If $\mathbf{W} = \text{Var}(\hat{\mathbf{e}})$, then the covariance matrix of the GLS estimator [@greeneeconometric] for the bottom level is given by:

$$\text{Var}(\tilde{\mathbf{b}})=(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}$$

Let $\mathcal{H}_k$ be a temporal hierarchy with a set of aggregation levels, characterized by the summing matrix $\mathbf{S}_k$ and base forecast covariance $\mathbf{W}_k$. For the sake of the proof, we define $\mathcal{I}_k$ to be the inverse of the bottom level's covariance matrix

$$\mathcal{I}_k=\text{Var}(\tilde{\mathbf{b}})^{-1}=\mathbf{S}_k'\mathbf{W}_k^{-1}\mathbf{S}_k$$

When we introduce a new intermediate aggregation level to create hierarchy $\mathcal{H}_{k+1}$, we are appending new observations (base forecasts) to the system. The augmented total covariance matrix $\mathbf{W}_{k+1}$, accounting for potential correlations between the existing and newly added base forecasts, can be written as a block matrix:

$$\mathbf{W}_{k+1}=\begin{bmatrix}\mathbf{W}_k&\mathbf{C}\\\mathbf{C}'&\mathbf{W}_{new}\end{bmatrix}$$

where $\mathbf{W}_{new}$ is the variance of the newly added forecasts, and $\mathbf{C}$ is the cross-covariance between the old and new forecasts. The total summing matrix is expanded accordingly:

$$\mathbf{S}_{k+1}=\begin{bmatrix}\mathbf{S}_k\\\mathbf{S}_{new}\end{bmatrix}$$

To calculate the the new inverse $\mathcal{I}_{k+1}=\mathbf{S}_{k+1}'\mathbf{W}_{k+1}^{-1}\mathbf{S}_{k+1}$, we invert the block covariance matrix $\mathbf{W}_{k+1}$ using the Schur Complement. Let $\mathbf{K}$ be the Schur Complement of $\mathbf{W}_k$ in $\mathbf{W}_{k+1}$:

$$\mathbf{K}=\mathbf{W}_{new}-\mathbf{C}'\mathbf{W}_k^{-1}\mathbf{C}$$

Because $\mathbf{W}_{k+1}$ is a positive definite covariance matrix, by the Haynsworth inertia additivity formula, its Schur Complement $\mathbf{K}$ is also positive definite, ensuring $\mathbf{K}^{-1}$ exists and is positive definite [@zhangschur].

Using standard block inversion theorems, the inverse of our total covariance matrix is perfectly defined as:

$$\mathbf{W}_{k+1}^{-1} = \begin{bmatrix} \mathbf{W}_k^{-1} + \mathbf{W}_k^{-1}\mathbf{C}\mathbf{K}^{-1}\mathbf{C}'\mathbf{W}_k^{-1} & -\mathbf{W}_k^{-1}\mathbf{C}\mathbf{K}^{-1} \\ -\mathbf{K}^{-1}\mathbf{C}'\mathbf{W}_k^{-1} & \mathbf{K}^{-1} \end{bmatrix}$$

Now we substitute this block inverse matrix back into our $\mathcal{I}_{k+1}=\mathbf{S}_{k+1}'\mathbf{W}_{k+1}^{-1}\mathbf{S}_{k+1}$ equation:

$$\mathcal{I}_{k+1} = \begin{bmatrix} \mathbf{S}_k' & \mathbf{S}_{new}' \end{bmatrix} \begin{bmatrix} \mathbf{W}_k^{-1} + \mathbf{W}_k^{-1}\mathbf{C}\mathbf{K}^{-1}\mathbf{C}'\mathbf{W}_k^{-1} & -\mathbf{W}_k^{-1}\mathbf{C}\mathbf{K}^{-1} \\ -\mathbf{K}^{-1}\mathbf{C}'\mathbf{W}_k^{-1} & \mathbf{K}^{-1} \end{bmatrix} \begin{bmatrix} \mathbf{S}_k \\ \mathbf{S}_{new} \end{bmatrix}$$

When this matrix multiplication is solved and and all the terms are expanded, five distinct terms remain:

1. $+ \mathbf{S}_k' \mathbf{W}_k^{-1} \mathbf{S}_k$
2. $+ \mathbf{S}_k' \mathbf{W}_k^{-1} \mathbf{C} \mathbf{K}^{-1} \mathbf{C}' \mathbf{W}_k^{-1} \mathbf{S}_k$
3. $- \mathbf{S}_k' \mathbf{W}_k^{-1} \mathbf{C} \mathbf{K}^{-1} \mathbf{S}_{new}$
4. $- \mathbf{S}_{new}' \mathbf{K}^{-1} \mathbf{C}' \mathbf{W}_k^{-1} \mathbf{S}_k$
5. $+ \mathbf{S}_{new}' \mathbf{K}^{-1} \mathbf{S}_{new}$

The first term is exactly $\mathcal{I}_{old}$. We can take the remaining four terms and perfectly factor them into a single, elegant quadratic equation. Let us define a new matrix $\mathbf{A}$:

$$\mathbf{A} = \mathbf{S}_{new} - \mathbf{C}'\mathbf{W}_k^{-1}\mathbf{S}_k$$

It is a valid matrix of shape $(m_{new} \times n)$, because the shape of $\mathbf{S}_{new}$ is $(m_{new} \times n)$, the shape of $\mathbf{S}_k$ is $(m_k \times n)$, the shape of $\mathbf{W}_k^{-1}$ is $(m_k \times m_k)$ and the shape of $\mathbf{C}$ is $(m_k \times m_{new})$.

We can also derive $\mathbf{A}'$ to be

$$\mathbf{A}' = (\mathbf{S}_{new} - \mathbf{C}'\mathbf{W}_k^{-1}\mathbf{S}_k)'$$

$$\mathbf{A}' = \mathbf{S}_{new}' - (\mathbf{S}_k' (\mathbf{W}_k^{-1})' (\mathbf{C}')')$$

Since $\mathbf{W}_k$ is symmetrical, its inverse is also

$$\mathbf{A}' = \mathbf{S}_{new}' - \mathbf{S}_k'\mathbf{W}_k^{-1}\mathbf{C}$$

Now if we calculate the matrix $\mathbf{A}' \mathbf{K}^{-1} \mathbf{A}$ (the shape of $\mathbf{K}^{-1}$ is $(m_{new} \times m_{new})$)

$$(\mathbf{S}_{new}' - \mathbf{S}_k'\mathbf{W}_k^{-1}\mathbf{C}) \mathbf{K}^{-1} (\mathbf{S}_{new} - \mathbf{C}'\mathbf{W}_k^{-1}\mathbf{S}_k)$$

$$(\mathbf{S}_{new}' - \mathbf{S}_k'\mathbf{W}_k^{-1}\mathbf{C}) (\mathbf{K}^{-1}\mathbf{S}_{new} - \mathbf{K}^{-1}\mathbf{C}'\mathbf{W}_k^{-1}\mathbf{S}_k)$$

From here, we can multiply the binomial brackets to give us exactly the terms 2.-5. in $\mathcal{I}_{k+1}$ Therefore, that entire formula collapses into:

$$\mathcal{I}_{k+1}=\mathcal{I}_k+\mathbf{A}'\mathbf{K}^{-1}\mathbf{A}$$

where $\mathbf{A}=\mathbf{S}_{new}-\mathbf{C}'\mathbf{W}_k^{-1}\mathbf{S}_k$. Because $\mathbf{K}^{-1}$ is positive definite, the quadratic form $\mathbf{A}'\mathbf{K}^{-1}\mathbf{A}$ is mathematically guaranteed to be a PSD matrix [@sebermatrix]. Adding a PSD matrix implies an inequality defined by the Loewner partial order:

$$\mathcal{I}_{k+1}\succeq\mathcal{I}_k$$

A theorem of the Loewner partial order states that for Hermitian matrices, inversion reverses the inequality [@sebermatrix]. A variance matrix $\text{Var}(\tilde{\mathbf{b}})$ is Hermitian because each of its components are real numbers, and it is symmetric. Since $\mathcal{I}$ is defined to be the inverse of $\text{Var}(\tilde{\mathbf{b}})$, it is also Hermitian, therefore the theorem does apply.

$$\mathcal{I}_k^{-1}\succeq\mathcal{I}_{k+1}^{-1}$$

$$\text{Var}(\tilde{\mathbf{b}}_k)\succeq\text{Var}(\tilde{\mathbf{b}}_{k+1})$$

This proves that incorporating intermediate temporal levels strictly produces a PSD reduction in the covariance of the bottom-level reconciled forecasts, regardless of the correlation structure of the base forecasts.

#### Generalization to aggregations

To get the reconciled forecasts across the hierarchy, we apply the summing matrix to the resulting bottom level of the reconciliation: $\tilde{\mathbf{y}}=\mathbf{S}_0\tilde{\mathbf{b}}$. In the research, only aggregation levels which were common across hierarchies were analyzed. Assume that the specific summing matrix to create these common aggregation levels is $\mathbf{S}_0$

$$\text{Var}(\tilde{\mathbf{y}}) = \text{Var}(\mathbf{S}_0\tilde{\mathbf{b}})$$

$$\text{Var}(\tilde{\mathbf{y}}) = \mathbf{S}_0 \text{Var}(\tilde{\mathbf{b}}) \mathbf{S}_0'$$

We have shown that $\text{Var}(\tilde{\mathbf{b}})$ is PSD reduced by incorporating intermediate levels, therefore $\mathbf{\Delta}$ is a PSD matrix given by

$$\mathbf{\Delta} = \text{Var}(\tilde{\mathbf{b}}_k) - \text{Var}(\tilde{\mathbf{b}}_{k+1}) \succeq 0$$

We define $\tilde{\mathbf{y}}_k$ and $\tilde{\mathbf{y}}_{k+1}$ to be the reconciled forecasts of the common aggregation levels, from the two hierarchies:

$$\tilde{\mathbf{y}}_k = \mathbf{S}_0 \tilde{\mathbf{b}}_k$$

$$\tilde{\mathbf{y}}_{k+1} = \mathbf{S}_0 \tilde{\mathbf{b}}_{k+1}$$

If we look at the difference of the variances of these common reconciled forecast levels

$$\text{Var}(\tilde{\mathbf{y}}_k) - \text{Var}(\tilde{\mathbf{y}}_{k+1}) = \mathbf{S}_0 \text{Var}(\tilde{\mathbf{b}}_k) \mathbf{S}_0' - \mathbf{S}_0 \text{Var}(\tilde{\mathbf{b}}_{k+1}) \mathbf{S}_0'$$

$$\text{Var}(\tilde{\mathbf{y}}_k) - \text{Var}(\tilde{\mathbf{y}}_{k+1}) = \mathbf{S}_0 \left( \text{Var}(\tilde{\mathbf{b}}_k) - \text{Var}(\tilde{\mathbf{b}}_{k+1}) \right) \mathbf{S}_0'$$

$$\text{Var}(\tilde{\mathbf{y}}_k) - \text{Var}(\tilde{\mathbf{y}}_{k+1}) = \mathbf{S}_0 \mathbf{\Delta} \mathbf{S}_0'$$

Because is $\mathbf{\Delta}$ is PSD, the quadratic form $\mathbf{S}_0 \mathbf{\Delta} \mathbf{S}_0'$ is also guaranteed to be PSD [@sebermatrix]. Therefore, we can write the following.

$$\text{Var}(\tilde{\mathbf{y}}_k) - \text{Var}(\tilde{\mathbf{y}}_{k+1}) \succeq 0$$

$$\text{Var}(\tilde{\mathbf{y}}_k) \succeq \text{Var}(\tilde{\mathbf{y}}_{k+1})$$

#### Application to RMSE

Because we have established that $\text{Var}(\tilde{\mathbf{y}}_k)\succeq\text{Var}(\tilde{\mathbf{y}}_{k+1})$, the extracted diagonal elements representing the variance of any specific individual aggregation node $i$ strictly cannot increase:

$$(\text{Var}(\tilde{\mathbf{y}}_k))_{ii}\ge(\text{Var}(\tilde{\mathbf{y}}_{k+1}))_{ii}$$

It follows directly that the Root Mean Squared Error (RMSE) for that node cannot increase, as $\text{RMSE}_i=\sqrt{V_{ii}}$. Furthermore, this monotonic non-increase holds true for the Normalised RMSE (NRMSE). Normalisation merely applies a strictly positive scalar divisor $c_i$ (the variance of the training data at that specific granularity) to the RMSE:

$$\text{NRMSE}_i=\frac{\sqrt{V_{ii}}}{c_i}$$

Since $c_i$ is a constant derived purely from historical training data and is completely independent of the reconciliation hierarchy, the inequality holds:

$$(\text{NRMSE}_k)_i\ge(\text{NRMSE}_{k+1})_i$$

Therefore, we guarantee in theory that adding any intermediate temporal levels to a hierarchy cannot increase the expected NRMSE at any aggregation node.

#### In practice

What was proven here is that for a single horizon of a base period (i.e. 28 days), the NRMSE of individual temporal aggregations do not increase. What was studied in practice is the NRMSE across a greater forecast horizon aggregated across different time series of different properties. While theoretical guarantees show that the expected NRMSE strictly does not increase for any individual aggregation, empirical applications revealed a divergence in summary statistics.

\newpage
