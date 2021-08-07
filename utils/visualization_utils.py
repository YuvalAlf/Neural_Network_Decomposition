import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def visualize_histogram(data, x_label, title):
    plt.figure()
    sns.kdeplot(data, color='g')
    plt.title(title)
    plt.xlabel(x_label)
    plt.show()
