## Conclusions

The paper studied whether aggregating the time series at more and more levels, and forecasting at each of them independently could improve the accuracy of predictions after reconciliation. We have found practical evidence that this could work well for top level granularity, regardless of what exact intermediate aggregations we might be using below our target. We found that at bottom level granularity this improvement is somewhat present, but its magnitude is very negligible. After sufficient practical evidence was gathered, the paper set out to mathematically prove what we have seen in practice, with concrete temporal hierarchies. The proven theory would suggest that the error metrics of all aggregation levels (top and bottom level included) should improve by having more aggregation levels. But this does not 100% hold up in practice, as it was said, at daily granularity the improvement of NRMSE was negligible. This can be attributed to the following aspects

- The covariance matrix of the base forecasts are only estimated, and for our use, the number of residuals were less then the number of variables, so only shrinkage method could be used to estimate the matrix, which is not ideal.
- The base forecasts are likely unbiased in most cases, but it's not guaranteed
- Bottom level granularity forecasts are stripped apart by the inner workings of temporal hierarchies, meaning, unfortunately minT has to look at the variance of 28 days independently

Nonetheless at top level - monthly granularity - we could show that the theory does hold up. We can assume that by improving RMSE metric of the errors, other error metrics are also likely to be improved, except in extreme cases.

This means that supposedly, this method of adding arbitrary aggregation levels below (or even above) our target aggregation, can meaningfully improve forecast accuracy, at the cost of computation time, and by satisfying the requirement of having to access some lower granularity level of the data (e.g. if we want to precisely forecast monthly data, we need to have some higher granularity resolution of the data, for example, daily, as it was in our case).

### Sources

- [Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2017). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization, Section 2.2](https://robjhyndman.com/papers/MinT.pdf)
- [George Athanasopoulosa, Rob J. Hyndmana, Nikolaos Kourentzesb, Fotios Petropoulosc. Forecasting with Temporal Hierarchies](https://robjhyndman.com/papers/temporalhierarchies.pdf)
- [Chan, Univ. of Sydney. Restricted Least Squares](https://www.maths.usyd.edu.au/u/jchan/GLM/RestrictedLeastSquares.pdf)
- [FoRecoPy documentation](https://danigiro.github.io/FoRecoPy/)
