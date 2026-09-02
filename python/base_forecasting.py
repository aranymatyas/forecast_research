from prophet import Prophet
from statsforecast.models import AutoETS, AutoARIMA, AutoTheta
from statsforecast import StatsForecast
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

STATS_MODELS = {
    'AutoETS': AutoETS,
    'AutoARIMA': AutoARIMA,
    'AutoTheta': AutoTheta,
}

OTHER_MODELS = {
    'Prophet': lambda: Prophet(uncertainty_samples=0)
}

ALL_MODELS = STATS_MODELS | OTHER_MODELS

# Minimum training observations at coarsest level (top level) to avoid model errors
MODEL_MIN_TRAIN_OBS = {
    'AutoETS': 11,
    'AutoARIMA': 4,
    'AutoTheta': 4,
    'Prophet': 2
}


def make_forecasts(X_train_wide: pd.DataFrame, model_str: str, h_steps: int, freq: str, n_jobs: int, verbose: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Makes forecasts with the specified model for each column of the wide format
    train data set.

    Returns:
        forecasts_wide, residuals_wide
    '''
    if model_str in STATS_MODELS:
        return _stats_forecast(X_train_wide, model_str, h_steps, freq, n_jobs, verbose)
    elif model_str in OTHER_MODELS:
        return _other_forecast(X_train_wide, model_str, h_steps, freq, n_jobs, verbose)
    else:
        raise ValueError(f"Unknown model: {model_str}. Available: {list(ALL_MODELS.keys())}")


def _stats_forecast(X_train_wide: pd.DataFrame, model_str: str, h_steps: int, freq: str, n_jobs: int, verbose: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' Makes forecasts if the model is one in the StatsForecast library '''
    model_cls = STATS_MODELS[model_str]
    # Convert to long format for batch forecasting
    df_long = X_train_wide.reset_index().melt(id_vars='ds', var_name='unique_id', value_name='y')
    df_long = df_long.dropna(subset=['y']).reset_index(drop=True)

    # Single batch call — parallelizes across series internally
    sf = StatsForecast(models=[model_cls()], freq=freq, n_jobs=n_jobs, verbose=verbose)
    forecast_df = sf.forecast(df=df_long, h=h_steps, fitted=True)
    fitted_df = sf.forecast_fitted_values()

    # Pivot forecasts back to wide
    forecast_df = forecast_df.reset_index()
    forecast_wide = forecast_df.pivot(index='ds', columns='unique_id', values=model_str)
    forecast_wide.columns.name = None

    # Compute and pivot residuals
    fitted_df = fitted_df.reset_index()
    fitted_df['residual'] = fitted_df['y'] - fitted_df[model_str]
    residual_wide = fitted_df.pivot(index='ds', columns='unique_id', values='residual')
    residual_wide.columns.name = None

    return forecast_wide, residual_wide


def _other_forecast(X_train_wide: pd.DataFrame, model_str: str, h_steps: int, freq: str, n_jobs: int, verbose: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' Makes forecasts if the model is one in some non StatsForecast library '''
    model_cls = OTHER_MODELS[model_str]

    # Prepare per-series DataFrames
    tasks = []
    for col in X_train_wide.columns:
        df = X_train_wide[[col]].reset_index()
        df.columns = ['ds', 'y']
        df = df.loc[df['y'].first_valid_index():].reset_index(drop=True)
        tasks.append((col, df, h_steps, freq))

    results = []
    with tqdm(total=len(tasks), desc="Forecast", disable=not verbose) as pbar:
        for result in Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_fit_single_series)(model_cls, col, df, h, freq)
            for col, df, h, freq in tasks
        ):
            results.append(result)
            pbar.update(1)

    # Collect results
    forecast_dict = {}
    residual_dict = {}
    for result in results:
        col, forecast_series, residual_series = result
        forecast_dict[col] = forecast_series
        residual_dict[col] = residual_series

    # Build wide DataFrames
    forecast_wide = pd.DataFrame(forecast_dict)
    forecast_wide.index.name = 'ds'

    residual_wide = pd.DataFrame(residual_dict)
    residual_wide.index.name = 'ds'

    return forecast_wide, residual_wide


def _fit_single_series(model_cls, col: str, df: pd.DataFrame, h: int, freq: str) -> tuple[str, pd.Series, pd.Series]:
    ''' Fits a single model on one series. Designed for parallel execution. '''
    model = model_cls()
    model.fit(df)

    future = model.make_future_dataframe(periods=h, freq=freq)
    forecast = model.predict(future)

    yhat = forecast.set_index('ds')['yhat']
    forecast_series = yhat.tail(h)
    residual_series = df.set_index('ds')['y'] - yhat.head(len(df))

    return col, forecast_series, residual_series