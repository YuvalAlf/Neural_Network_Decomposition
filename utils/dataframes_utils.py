import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def filter_dataset(df, columns_to_keep):
    return df[columns_to_keep]


def read_dataset(dataset_path):
    return pd.read_csv(dataset_path)

