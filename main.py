from typing import List

from pandas import DataFrame

from utils.data_frame_utils import read_dataset, filter_data_frame, add_log_column
from utils.functional_utils import run_on_true
from utils.normalizer import DataFrameNormalizer
from utils.visualization_utils import visualize_histogram, visualize_scatter_plot, visualize_correlation_matrix


def read_houses_prices_dataset(path: str) -> (List[str], str, DataFrame):
    data_frame = read_dataset(path)
    x_columns = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF',
                 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'YrSold']
    y_column = 'SalePrice'
    filtered_data_frame = filter_data_frame(data_frame, *x_columns, y_column)
    return x_columns, y_column, filtered_data_frame


def main():
    x_columns, sale_price, houses_prices = read_houses_prices_dataset('datasets/houses_prices.csv')
    log_sale_price = add_log_column(houses_prices, sale_price)

    for y_column in [sale_price, log_sale_price]:
        # run_on_true(True, lambda: visualize_correlation_matrix(houses_prices, y_column))
        # run_on_true(True, lambda: visualize_histogram(houses_prices[y_column].values, y_column, 'Houses Prices'))
        # for x_column in x_columns:
        #     run_on_true(True, lambda: visualize_scatter_plot(houses_prices, x_column, y_column))

        normalized_data_frame, normalizer = DataFrameNormalizer.normalize_data_frame(houses_prices)
        # for inner_layer_sizes in ([12],):
        #     reg_nn = RegressionNetwork(len(x_columns), inner_layer_sizes, 1)
        #     name = f"{y_column}_{mkstring(reg_nn.layers, '[', ',', ']')}"
        #     median_score = reg_nn.train(houses_prices_df[x_columns].values, houses_prices_df[y_column].values, name)


if __name__ == '__main__':
    main()
