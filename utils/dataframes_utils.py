import pandas as pd


def filter_dataset(df, columns_to_keep):
    return df[columns_to_keep]


def read_dataset(dataset_path):
    return pd.read_csv(dataset_path)

