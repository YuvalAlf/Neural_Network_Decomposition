import numpy as np
import pandas as pd
from pandas import DataFrame


def filter_data_frame(data_frame: DataFrame, *columns_to_keep: str) -> DataFrame:
    return data_frame[list(columns_to_keep)]


def read_dataset(dataset_path: str) -> DataFrame:
    return pd.read_csv(dataset_path)


def normalize_dataframe(data_frame: pd.DataFrame) -> DataFrame:
    normalized_data_frame = data_frame.copy()
    for column in data_frame.columns:
        normalized_data_frame[column] = (data_frame[column] - data_frame[column].mean()) / data_frame[column].std()
    return normalized_data_frame


def add_log_column(data_frame: pd.DataFrame, column_name: str) -> str:
    log_column_name = f'Log({column_name})'
    data_frame[log_column_name] = np.log(data_frame[column_name])
    return log_column_name
