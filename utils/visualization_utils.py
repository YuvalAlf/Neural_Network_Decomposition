import os
from typing import Callable, Any, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from globals import CMAP, COLOR, FIG_SIZE, KDE_COLOR, SCATTER_COLOR
from utils.functional_utils import curry, apply_to_result


@apply_to_result('_'.join)
def extract_fig_name(plot_function_name: str, *args: Any) -> None:
    if plot_function_name.startswith('visualize_'):
        yield plot_function_name[len('visualize_'):]
    else:
        yield plot_function_name
    for arg in args[1:]:
        if not isinstance(arg, DataFrame):
            yield str(arg)


@curry
def visualize_plot(fill_plot_function: Callable[..., None], *args: Any) -> None:
    plt.figure(figsize=FIG_SIZE)
    fill_plot_function(*args)
    plt.tight_layout()
    fig_name = extract_fig_name(fill_plot_function.__name__, *args)
    plt.savefig(os.path.join('plots', f'{fig_name}.pdf'), bbox_inches='tight')
    # plt.show()
    plt.close()


@visualize_plot
def visualize_histogram(kde_data: Sequence[float], x_label: str, title: str) -> None:
    sns.kdeplot(kde_data, color=KDE_COLOR)
    plt.title(title)
    plt.xlabel(x_label)


@visualize_plot
def visualize_scatter_plot(data_frame: DataFrame, x_entry: str, y_entry: str) -> None:
    sns.scatterplot(data=data_frame, x=x_entry, y=y_entry, alpha=0.5, color=SCATTER_COLOR)
    plt.xlabel(x_entry)
    plt.ylabel(y_entry)


@visualize_plot
def visualize_correlation_matrix(data_frame: DataFrame, y_column: str) -> None:
    sns.heatmap(data_frame.corr(), cmap=CMAP, linewidth=1, vmin=-1, vmax=1, annot=True, fmt='.2f')


@visualize_plot
def plot_predictions(predicted, expected, name):
    plt.scatter(expected, predicted, color=COLOR, alpha=0.5)
    min_val, max_val = min(expected), max(expected)
    plt.plot([min_val, max_val], [min_val, max_val], color='yellow', alpha=0.8)
    plt.savefig(f'plots/{name}.pdf', bbox_inches='tight')


@visualize_plot
def visualize_train_stats(stats):
    plt.plot(stats.history['loss'], color='green', alpha=0.8)

    plt.plot(stats.history['val_loss'], color='red', alpha=0.8)
    plt.legend(['train data', 'validation data'])
    plt.title('model accuracy')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
