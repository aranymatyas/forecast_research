import pandas as pd
import kagglehub
from datetime import datetime, timedelta
from typing import Literal
import zipfile

def fetch_dataset_m4(format: Literal["long", "wide"] = "wide", required_days: int = 31) -> pd.DataFrame:
    """ Return train dataset of M4 competition daily time series. Small dataset. Artificial dates. All time series shifted to bottom, lots missing. """
    path = kagglehub.dataset_download("yogesh94/m4-forecasting-competition-dataset", output_dir="data/m4")
    X = pd.read_csv(path + '/Daily-train.csv')

    # We have no real dates so assume some start
    start_date = datetime(2000, 1, 1)
    dates = pd.Series([start_date + timedelta(days=i) for i in range(len(X))])
    X.insert(0, 'ds', dates)
    X = X.drop(columns=['V1'])
    mask = X.tail(required_days).notna().all()

    X = X[mask[mask].index.tolist()]
    X = move_to_float32(X)

    if format == "long":
        X = wide_to_long(X)
    return X

def fetch_dataset_web_traffic(format: Literal["long", "wide"] = "wide"):
    ''' Very large dataset with significant amount of missing values '''
    with zipfile.ZipFile("data/web-traffic/train_2.csv.zip", 'r') as zip_ref, \
         zip_ref.open("train_2.csv") as f:
        X = pd.read_csv(f).dropna(axis=1, how='all')

    X = move_to_float32(X)
    X = X.rename(columns={'Page': 'ds'}).set_index('ds').T
    X = X.reset_index().rename(columns={'index': 'ds'})
    X.columns.name = None


    if format == "long":
            X = wide_to_long(X)

    return X

def fetch_dataset_corpo(format: Literal["long", "wide"] = "wide"):
    ''' Very large dataset with lots of missing values '''
    with zipfile.ZipFile("data/corpo/train.csv.zip", 'r') as zip_ref, \
         zip_ref.open("train.csv") as f:
        X = pd.read_csv(f)

    X = move_to_float32(X)
    X['unique_id'] = X['store_nbr'].astype(str) + '_' + X['item_nbr'].astype(str)

    X = X.rename(columns={'date': 'ds', 'unit_sales': 'y'})[['ds', 'unique_id', 'y']]
    X['ds'] = pd.to_datetime(X['ds'])


    if format == "wide":
        X = long_to_wide(X)
    return X

def fetch_dataset_m5(format: Literal["long", "wide"] = "wide"):
    ''' Dataset with no missing values. Many small values. Artificial dates. '''
    with zipfile.ZipFile("data/m5/sales_train_evaluation.csv.zip", 'r') as zip_ref, \
         zip_ref.open("sales_train_evaluation.csv") as f:
        X = pd.read_csv(f)

    X = move_to_float32(X)
    X = X.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    X = X.set_index('id').T #.reset_index().rename(columns={'index': 'ds'})
    X = X.reset_index().rename(columns={'index': 'ds'})
    X.columns.name = None

    # We have no real dates so assume some start
    start_date = datetime(2000, 1, 1)
    X['ds'] = pd.Series([start_date + timedelta(days=i) for i in range(len(X))])

    if format == "long":
        X = wide_to_long(X)
    return X


def move_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    float64_cols = df.select_dtypes(include=['float64']).columns
    return df.astype({col: 'float32' for col in float64_cols})

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt(
                id_vars=['ds'],
                var_name='unique_id',
                value_name='y'
            ).dropna(subset=['y']).reset_index(drop=True)

def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.pivot(index='ds', columns='unique_id', values='y').reset_index()
    df.columns.name = None
    return df

def get_first_and_last_date(df: pd.DataFrame):

    if 'unique_id' in df.columns:
        # Long Format

        value_col = 'y'

        valid_df = df.dropna(subset=[value_col])

        res = valid_df.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
        res.columns = ['unique_id', 'first_valid', 'last_valid']
        return res

    else:
        temp = df.set_index('ds')

        first_date = temp.apply(lambda row: row.first_valid_index(), axis=0)
        last_date = temp.apply(lambda row: row.last_valid_index(), axis=0)

        res = pd.DataFrame({
            'unique_id': first_date.index,
            'first_valid': first_date.values,
            'last_valid': last_date.values
        })
        return res
