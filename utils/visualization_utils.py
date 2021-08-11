from functools import wraps

import matplotlib.pyplot as plt
import seaborn as sns

from globals import VISUALIZE, CMAP, COLOR


def visualize_plot(func):
    @wraps(func)
    def returned_function(*args, **kwargs):
        if VISUALIZE:
            plt.figure(figsize=[12, 8])
            ret_value = func(*args, **kwargs)
            plt.tight_layout()
            # plt.show()
            plt.close()
            return ret_value
    return returned_function


@visualize_plot
def visualize_histogram(data, x_label="", title=""):
    sns.kdeplot(data, color=COLOR)
    plt.title(title)
    plt.xlabel(x_label)
    plt.savefig(f'plots/{x_label}_{title}.pdf', bbox_inches='tight')


@visualize_plot
def visualize_scatter_plot(dataframe, x_entry, y_entry):
    sns.scatterplot(data=dataframe, x=x_entry, y=y_entry, alpha=0.5, color=COLOR)
    plt.xlabel(x_entry)
    plt.ylabel(y_entry)
    plt.savefig(f'plots/{y_entry}_{x_entry}.pdf', bbox_inches='tight')


@visualize_plot
def visualize_correlation_matrix(dataframe, name):
    sns.heatmap(dataframe.corr(), cmap=CMAP, linewidth=1, vmin=-1, vmax=1, annot=True, fmt='.2f')
    plt.savefig(f'plots/{name}_Correlation_Matrix.pdf', bbox_inches='tight')
