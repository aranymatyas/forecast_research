## Abstract

The fundamental goal of the paper is to see whether we can produce more accurate forecasts for a time series at both daily and monthly level by leveraging temporal hierarchical forecast reconciliation. Temporal hierarchical forecast reconciliation is a process where we forecast at different granularities of a single time series at a fixed horizon (e.g. daily and weekly for 7 days), then reconcile those forecasts to produce coherent forecasts (e.g. the daily forecasts add up to the weekly).

Throughout this research, we exclusively evaluate the accuracy of forecasts at two target granularities: daily and monthly. However, we will leverage forecasts at intermediate aggregation levels (e.g. weekly) to try and improve the daily and monthly predictions. The objective is to explore how different intermediate aggregation levels can most effectively improve our target forecasts by using reconciliation.

### Assumptions, abilities and limitations

Throughout the project, the biggest limitation set, is that a month is defined as 28 days exactly. This decision was done to avoid unnecessary coding complexity concerning varying months lengths, which would not even yield any practical usefulness to the research.

We are forecasting for a horizon of 3 times 28 days, so roughly three months. This ensures that the horizon isn't so great on a daily level, but also not singular on monthly level. Hopefully this will be a nice middle ground. Accordingly, this much of the dataset will be cut off from the front of it, to serve as validation data, and the rest will be kept as training set of the base forecast models.

### Temporal Reconciliation methodology

The exact workings of temporal hierarchical forecast reconciliation can be read from [Forecasting with Temporal Hierarchies by Hyndman et al.](https://robjhyndman.com/papers/temporalhierarchies.pdf), and they will not be explained in depth here.

There were no existing libraries available which could perfectly satisfy my needs in a convenient way. I wanted to be able to define aggregations such as weekends and weekdays, which do not continiously use up all daily data, but rather skip some. To my understand there is no temporal reconciliation library in either python or R which would allow this.

Consequently I made my own small python framework for this project. It allows seemless aggregation, forecasting and reconciliation for any kind of aggregation, defined either by some periodicity, or by a row in the summing matrix. For this to work, there had to be a constraint introduced which meant that I needed to introduce a so called `Base Period`. This period defined how many columns the summing matrix will have, i.e. what will be a common period for all aggregations (relative to the granularity of the bottom level). I chose this number to be 28 for our use case, meaning the biggest aggregation I could have was semi-monthly (all mentions of monthly aggregation from now on will mean this 28-day period, as established in the limitations section), but below that top level, there could be any groupings of days defined and used.

By this, reconciliation could happen knowing that no matter what exact aggregations are set, the summing matrix will have 28 columns, and because of that, forecast horizons will always be multiples of this `Base Period`. This allowed me to use `Foreco` or its python port `Forecopy` to simply reconcile my forecasts, as if they were regular hierarchical forecasts, not temporal. The framework does this by flattening out the forecasts to match the summing matrix exactly, to basically abandon the idea of temporality. Take the situation where we have monthly, weekly and daily forecasts at a horizon of $3 \times 28$ days: I will have 3 monthly forecasts, 12 weekly, and 84 daily ones. Simply these forecasts do not allow us to calculate covariance, since there are different amounts of forecasts. So instead we flatten these so that we will have more virtual variables, with exactly the same amount of (samples): 1 row of forecasts will contain the following values as a flat vector: (1 monthly | 4 weekly | 28 daily), essentially transforming the 12 weekly forecasts to 3 sets of forecasts for $week_1, week_2, week_3, week_4$.

The advantage of this setup is that for periodic aggregations we can fit a base forecast model using the whole continiously aggregated history, which is only broken up/flattened afterwards, allowing for likely better predictions.

This setup required a few traits from the input data: the length of the time series shall be a multiple of 28 to allow for seemless aggregation of the data as preprocessing. After aggregating the time series and determining the forecast horizon as $h$ days, we cut off that amount of days from the end of the dataset, for validation, leaving the rest for training. Base forecast models (in our case AutoETS) are fit at each hierarchy levels on appropriate aggregations of the train set. Then by inference we retrieve appropriate amounts of predictions at each hierarchy level, which shall equate to $h$ days of data.

Afterwards, using the predictions and the validation data (which line up perfectly) some error metrics are calculated for each time series and hierarchy pair and are stored in a matrix. This is the only persistent data in the project, the actual forecasts are discarded after this calculation. The four error metrics calculated are the following:

- `NRMSE`: Normalised Root Mean Square Error. Normalization happens by division with the variance of the train set
- `WAPE`: Weighted Absolute Percentage Error. Weighing happens by the mean of the validation set
- `MASE`: Mean Absolute Scaled Error.  The scale is relative to a naive model's performance on the train set
- `1-R2`: One Minus $R^2$. The reason the standard $R^2$ is inversed, is that this way all error metrics share the property that 0 is their minimum possible value, which is also their best possible value.

### Analyis Methodology

For most of the paper, the `NRMSE` error metrics of forecasts will be analysed. The error metric will be calculated from the performance of the base forecasts and reconciled forecasts against a validation set.

The workflow was the following

---

1. Choosing dataset and filtering out a subset of it for faster computation

The dataset chosen for this was a Wikipedia Web traffic dataset, which will be described in depth later.

2. Train and validation set

In the way explained before, I cut of a portion of the historical data to serve as valiation set, and kept the rest as training data.

3. Defining hierarchies

Each section will look at the performance of different temporal hierarchies, these will be well defined at the beginning of sections

4. Base Forecasts

I used my small framework to make base predictions for each aggregation level in each hierarchy. The forecast model used for these independent base forecasts is AutoETS.

5. Reconciliation

I used shrinkage based minT reconciliation, in all hierarchies to have the base forecasts reconciled, independently per hierarchy. The reason I chose shrinkage method, is because this dataset did not provide sufficient residuals for more sophisticated covariance approximation methods, so this was the most advanced one, which I also read to perform quite well in research papers.

6. Error calculation

I then used my small framework to calculate 4 distinct error metrics of these base and reconciled forecasts, for each hierarchy, and both daily and monthly granularity. I then saved these error metrics, and discarded the actual forecasts, as they are not needed.

7. Error Analysis

Out of the 4 different error metrics, I will first analyse the normalized root mean square error metrics in this notebook. The normalisation happens by dividing the RMSE of the forecasts by the variance of the train set, to make the errors comparable across the various different time series. The other error metrics will be addressed concisely later on as well.

#### Dictionary

Some useful terms to list in advance, for easier comprehension. These will be used throughout the paper

| Term | Definition |
| --- | --- |
| Daily/Monthly granularity | The temporal resolution/aggregation level at which errors are evaluated. Daily refers to per-day forecast errors, monthly refers to per-28-day-period forecast errors. |
| Base forecast | The independent forecast produced by AutoETS at a given aggregation level, before any reconciliation is applied. |
| Reconciled forecast | The forecast obtained after applying minT shrinkage reconciliation within a given hierarchy, ensuring coherence across aggregation levels. |
| Base forecast error / Reconciled forecast error | The NRMSE error of a base or reconciled forecast, computed per time series at a given granularity. Each time series yields one error value. |
| Error difference (vs. base) | The difference: reconciled forecast error minus base forecast error, for a given hierarchy and granularity. A positive value means the error increased (worsened) after reconciliation; a negative value means it decreased (improved). |
| Positive / Negative error difference | The error difference filtered for only positive values (error increases) or only negative values (error improvements), used to analyze the two directions separately. |
| Hierarchy difference | The difference: `Semantic` reconciled error minus `Minimal` reconciled error, per time series. A negative value means `Semantic` achieved a lower error. Used in the Semantic vs. Minimal section. |
| Error sign distribution | A joint distribution showing the signs (+/-) of error differences at daily and monthly granularity simultaneously, yielding four categories: (+,+), (-,-), (+,-), (-,+). |
| Outlier | A forecast error in the top 5% of values (per error row). Outlier analyses use log-transformed values to make extreme values comparable. |
| Extreme outlier | A very rare, very large outlier that distorts mean and variance statistics. |
| Base inconsistency | The difference between the monthly base forecast and the sum of the corresponding daily base forecasts. Measures how incoherent the base forecasts are before reconciliation. |
| QQ plot | A quantile-quantile plot comparing two error distributions. Points on the identity line mean the distributions match; deviations show where one distribution has systematically larger or smaller values. |
| Percentile / Quantile (e.g. p75, p90) | The value below which a given percentage of error values fall. Used throughout to compare distributions without sensitivity to outliers. |
| Trend | A pattern observed when ordering the four hierarchies by the number of aggregation levels they contain (Minimal < Semantic < Binary < Full). When a metric changes monotonically across this ordering, we call it a trend. |

#### Python framework

The Python framework used in this research consists of 8 modules:

- `aggregation.py` — Defines temporal aggregation levels (Days, Weeks, Weekdays, Weekends, Fortnights, Months) and generates summing matrices. Each aggregation knows how to resample a wide-format time series dataset to its level. Any kind of additional arbitrary aggregation can be defined with its tools, as long as its interval fits in a 28 day window.
- `temporal_hierarchy.py` — Defines a `TemporalHierarchy` as an ordered list of aggregation levels. Handles resampling, base forecasting (via AutoETS through StatsForecast), storing reconciled forecasts, and producing the full summing matrix. Predefined hierarchies include `Minimal`, `Semantic`, and `Binary`. Any kind of arbitrary hierarchies may be defined by determining the aggregation levels.
- `time_series.py` — A thin wrapper around wide-format DataFrames, tracking the aggregation level and providing a flattened view for reconciliation tools.
- `forecast_reconciliation.py` — Wraps the `forecopy` (foreco) library to perform cross-sectional minT reconciliation on the flattened base forecasts and residuals. Also computes base forecast inconsistency.
- `error_metrics.py` — Computes NRMSE, MASE, WAPE, and 1-R² per time series at any granularity. Provides composable wrappers: `FilteredErrorMetrics`, `MergedErrorMetrics`, `ErrorDifference`. `MetricsIndex` enables concise and powerful multi-level indexing into the metrics table. This basically is an abstraction of a pandas table, to avoid dealing with it directly.
- `plots.py` — Plotly-based visualization functions: error histograms, QQ plots, error difference scatter plots, error-vs-error scatter/heatmaps, box/strip plots, error sign heatmaps, and error-over-inconsistency plots.
- `utils.py` — Dataset loaders (Wikipedia Web Traffic, M4, M5, Corporación Favorita), wide/long format conversions, series alignment/trimming to 28-day multiples, and dataset reduction helpers.
- `sampled_hierarchies.py` — This module allows to look at the statistics of many different hierarchies at once, it will be explained in depth later on.

In this final draft, code snippets will not be shown for less confusion. Every plot and table will be explained well, to account for this loss.

#### Dataset preprocessing

The dataset I am using in this paper is Google's [Web Traffic Time Series Forecasting](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/data). In its raw form it has 145k independent time series describing the web traffic of various wikipedia pages. Due to the nature of the data source, there are many different kinds of time series with different properties, hence why I chose this.

Additional reasons for why the dataset was chosen:

- Daily granularity: this was of course a prerequisite for this project.
- Has exact date timestamp: some datasets are only a sequence of values, not labeled by actual timestamps. This is convenient because we want to aggregate by weekdays and weekends too.
- High variance in data: unlike commercial datasets, this dataset rarely has long intervals of constant zero values.
- Big volume of time series: although for this analysis we are only using 10% of available data, it's still useful to be able to extend this any time.

I filtered out the dataset to only keep those time series, which have no missing data in their entire history, this way 115k were left. This was still way to much computationally to work with on a laptop, so I picked the first 13k time series as my dataset.

I trimmed the data to start on a Monday and to have its length be a multiple of 28. After trimming, 784 days were left in each time series, which equates to 112 weeks, so just over two years. This is still a satisfactory amount.

\newpage
