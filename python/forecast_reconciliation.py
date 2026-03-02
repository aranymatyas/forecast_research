import pandas as pd
import numpy as np
import forecopy as foreco

from python.temporal_hierarchy import TemporalHierarchy

class ForecastReconciliation:
    def __init__(self, temporal_hierarchy: TemporalHierarchy):
        self.temporal_hierarchy = temporal_hierarchy

    def get_agg_mat(self):
        agg_mat = self.temporal_hierarchy.make_summing_matrix(include_hierarchy_names=True)
        agg_mat = agg_mat[agg_mat['hierarchy'] != 'Days'] # Drop identity part
        return agg_mat.drop(columns=['hierarchy']).values

    def reconcile(self, method: str = 'ols'):
        try:
            agg_mat = self.get_agg_mat()
            base_forecasts = self.temporal_hierarchy.reorder_base_forecasts()
            residuals = self.temporal_hierarchy.reorder_residuals()
        except Exception as e:
            print("Base forecasts have probably not been produced")
            raise e

        # Cross sectional reconciliation
        reconciled_forecasts = np.empty_like(base_forecasts)
        for base_forecast, residual, i in zip(base_forecasts, residuals, range(len(base_forecasts))):
            reconciled_forecast = foreco.csrec(base=base_forecast, res=residual, agg_mat=agg_mat, comb=method)
            reconciled_forecasts[i] = reconciled_forecast

        self.temporal_hierarchy.store_reconciled_forecasts(reconciled_forecasts)
        return reconciled_forecasts
