import numpy as np
import forecopy as foreco
from typing import Literal

from python.temporal_hierarchy import TemporalHierarchy
from python.time_series import BASE_FORECASTS, RESIDUALS

class ForecastReconciliation:
    def __init__(self, temporal_hierarchy: TemporalHierarchy):
        self.temporal_hierarchy = temporal_hierarchy

    def _get_agg_mat(self):
        agg_mat = self.temporal_hierarchy.make_summing_matrix(include_hierarchy_names=True)
        agg_mat = agg_mat[agg_mat['hierarchy'] != 'Days'] # Drop identity part
        return agg_mat.drop(columns=['hierarchy']).values

    def reconcile(self, method: Literal['ols', 'str', 'wls', 'shr', 'sam'] = 'ols'):
        try:
            agg_mat = self._get_agg_mat()
            base_forecasts = self.temporal_hierarchy[BASE_FORECASTS]
            residuals = self.temporal_hierarchy[RESIDUALS]
        except Exception as e:
            print("Base forecasts have probably not been produced")
            raise e

        # Cross sectional reconciliation
        # Loop over time series and forecast horizons
        reconciled_forecasts = np.empty_like(base_forecasts)
        for ts_idx in range(base_forecasts.shape[0]):  # each time series
            for h_idx in range(base_forecasts.shape[1]):  # each forecast horizon
                base = base_forecasts[ts_idx, h_idx, :]  # one vector across hierarchy
                res = residuals[ts_idx, :, :]

                # Drop rows with any NaN values
                valid_rows = ~np.isnan(res).any(axis=1)
                res_clean = res[valid_rows]

                reconciled = foreco.csrec(base=base, res=res_clean, agg_mat=agg_mat, comb=method)
                reconciled_forecasts[ts_idx, h_idx, :] = reconciled

        self.temporal_hierarchy.store_reconciled_forecasts(reconciled_forecasts, method=method)
