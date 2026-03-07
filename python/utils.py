import pandas as pd
import numpy as np
import kagglehub
from datetime import datetime, timedelta, date
from typing import Literal
import zipfile

type TimeSeriesFormat = Literal["long", "wide"]

def fetch_dataset_m4(format: TimeSeriesFormat = "wide") -> pd.DataFrame:
    """ Return train dataset of M4 competition daily time series. Small dataset. Artificial dates. All time series shifted to bottom, lots missing. """
    path = kagglehub.dataset_download("yogesh94/m4-forecasting-competition-dataset", output_dir="data/m4")
    X = pd.read_csv(path + '/Daily-train.csv')

    # We have no real dates so assume some start
    X = move_to_float32(X)
    start_date = datetime(2000, 1, 1)
    dates = pd.Series([start_date + timedelta(days=i) for i in range(len(X))])
    X.insert(0, 'ds', dates)
    X = X.set_index('ds')
    X = X.drop(columns=['V1'])


    if format == "long":
        X = wide_to_long(X)
    return X

def fetch_dataset_web_traffic(format: TimeSeriesFormat = "wide"):
    ''' Very large dataset with significant amount of missing values '''
    with zipfile.ZipFile("data/web-traffic/train_2.csv.zip", 'r') as zip_ref, \
         zip_ref.open("train_2.csv") as f:
        X = pd.read_csv(f).dropna(axis=1, how='all')

    X = move_to_float32(X)
    X = X.rename(columns={'Page': 'ds'}).set_index('ds').T
    X.index.name = 'ds'
    X.index = pd.to_datetime(X.index)
    X.columns.name = None


    if format == "long":
            X = wide_to_long(X)

    return X

def fetch_dataset_corpo(format: TimeSeriesFormat = "wide"):
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

def fetch_dataset_m5(format: TimeSeriesFormat = "wide"):
    ''' Dataset with no missing values. Many small values. Real dates from 2011-01-29 to 2016-06-19. '''
    with zipfile.ZipFile("data/m5/sales_train_evaluation.csv.zip", 'r') as zip_ref, \
         zip_ref.open("sales_train_evaluation.csv") as f:
        X = pd.read_csv(f)
    
    # Load calendar to get real dates
    with zipfile.ZipFile("data/m5/calendar.csv.zip", 'r') as zip_ref, \
         zip_ref.open("calendar.csv") as f:
        calendar = pd.read_csv(f)
    
    X = move_to_float32(X)
    X = X.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    X = X.set_index('id').T
    X = X.reset_index().rename(columns={'index': 'ds'})
    X.columns.name = None
    
    # Map d_1, d_2, etc. to real dates using calendar
    date_mapping = dict(zip(calendar['d'], pd.to_datetime(calendar['date'])))
    X['ds'] = X['ds'].map(date_mapping)
    X = X.set_index('ds')

    if format == "long":
        X = wide_to_long(X)
    return X


def move_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    float64_cols = df.select_dtypes(include=['float64']).columns
    return df.astype({col: 'float32' for col in float64_cols})

def is_long(df: pd.DataFrame) -> bool:
    return 'unique_id' in df.columns

def is_wide(df: pd.DataFrame) -> bool:
    return not is_long(df)

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if 'ds' not in df.columns:
        df = df.reset_index()
    return df.melt(
                id_vars=['ds'],
                var_name='unique_id',
                value_name='y'
            ).dropna(subset=['y']).reset_index(drop=True)

def long_to_wide(df: pd.DataFrame, value_col: str = 'y') -> pd.DataFrame:
    df = df.pivot(index='ds', columns='unique_id', values=value_col)
    df.columns.name = None
    return df

def get_first_and_last_date(df: pd.DataFrame):
    ''' Returns the first and last date in each time series '''
    if is_long(df):
        # Long Format
        valid_df = df.dropna(subset=['y'])
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

def filter_last_days_available(df: pd.DataFrame, n_days: int = 31) -> pd.DataFrame:
    ''' Filter the dataset to only include series that have at least n_days of data available at the end. '''
    if is_long(df):
        # Long Format
        last_dates = df.groupby('unique_id')['ds'].max()
        mask = df.groupby('unique_id')['ds']
        valid_ids = last_dates[last_dates > (df['ds'].max() - timedelta(days=n_days))].index
        return df[df['unique_id'].isin(valid_ids)]

    else:
        # Wide format
        mask = df.tail(n_days).notna().all()
        return df[mask[mask].index.tolist()]

def reduce_dataset(df: pd.DataFrame, n_series: int = 100) -> pd.DataFrame:
    ''' Reduce the dataset to n_series variables '''
    if is_long(df):
        # Long Format
        unique_ids = df['unique_id'].unique()[:n_series]
        return df[df['unique_id'].isin(unique_ids)].reset_index(drop=True)
    else:
        # Wide Format
        cols_to_keep = df.columns[0:n_series].tolist()
        return df[cols_to_keep]



def get_nearest_monday(after: datetime) -> date:
    ''' Returns the first monday date after the given one '''
    days_to_monday = 7 - after.weekday()
    return (after + timedelta(days=days_to_monday % 7)).date()

def align_fill_and_trim_series(df: pd.DataFrame) -> pd.DataFrame:
    ''' Preprocess time series in wide format so that each hole is filled and each series is trimmed,
        so that the number of days are a multiple of 28, and each of them end on a Sunday
    '''
    conversion =  is_long(df)
    if conversion:
        df = long_to_wide(df)

    # Time series must end on sunday
    last_date = df.index.max()
    days_to_subtract = (last_date.dayofweek - 6) % 7
    end_sunday = last_date - timedelta(days=days_to_subtract)
    df_aligned = df.loc[:end_sunday]

    df_aligned = df_aligned.ffill()

    for col in df_aligned.columns:
        valid_mask = df_aligned[col].notna()
        valid_count = valid_mask.sum()

        remainder = valid_count % 28

        if remainder > 0:
            #Trim off start so time series is divisible by 28 days
            first_valid_dates = df_aligned[valid_mask].index[:remainder]
            df_aligned.loc[first_valid_dates, col] = np.nan

    df_aligned = df_aligned.dropna(axis=0, how='all')

    return df_aligned if not conversion else wide_to_long(df_aligned)
