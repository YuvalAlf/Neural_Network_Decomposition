import numpy as np

from utils.dataframes_utils import read_dataset, filter_dataset
from utils.visualization_utils import visualize_histogram, visualize_scatter_plot, visualize_correlation_matrix


def main():
    sale_price = 'SalePrice'
    log_sale_price = 'Log(SalePrice)'
    sale_price_column = 'SalePrice'
    x_columns = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF',
                       'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'YrSold']
    houses_prices_df = read_dataset('datasets/houses_prices.csv')
    houses_prices_df = filter_dataset(houses_prices_df, x_columns + [sale_price_column])
    houses_prices_df[log_sale_price] = np.log(houses_prices_df[sale_price])

    for y_column in [sale_price, log_sale_price]:
        visualize_correlation_matrix(houses_prices_df, y_column)
        visualize_histogram(houses_prices_df[y_column].values, y_column, 'Houses Prices')
        for x_column in x_columns:
            visualize_scatter_plot(houses_prices_df, x_column, y_column)
        # for build_options in (1,);
        #     nn = build_full_connected_nn(*build_options)


if __name__ == '__main__':
    main()
