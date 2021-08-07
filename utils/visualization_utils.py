from functools import wraps

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def visualize_plot(func):
    @wraps(func)
    def returned_function(*args, **kwargs):
        plt.figure()
        ret_value = func(*args, **kwargs)
        plt.tight_layout()
        plt.show()
        return ret_value
    return returned_function


@visualize_plot
def visualize_histogram(data, x_label="", title=""):
    sns.kdeplot(data, color='g')
    plt.title(title)
    plt.xlabel(x_label)


@visualize_plot
def visualize_scatter_plot(dataframe, x_entry, y_entry):
    sns.scatterplot(data=dataframe, x=x_entry, y=y_entry, alpha=0.5, color='g')
    plt.xlabel(x_entry)
    plt.ylabel(y_entry)
