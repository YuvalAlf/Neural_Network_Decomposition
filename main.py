import matplotlib.pyplot as plt
import numpy as np
from utils.dataframes_utils import read_dataset, filter_dataset
from utils.visualization_utils import visualize_histogram, visualize_scatter_plot


def main():
    sale_price = 'SalePrice'
    log_sale_price = 'Log(SalePrice)'
    columns_to_keep = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF',
                       'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'YrSold', 'SalePrice']
    houses_prices_df = read_dataset('datasets/houses_prices.csv')
    houses_prices_df = filter_dataset(houses_prices_df, columns_to_keep)
    houses_prices_df[log_sale_price] = np.log(houses_prices_df[sale_price])

    for y_column in [sale_price, log_sale_price]:
        visualize_histogram(houses_prices_df[y_column].values, y_column, 'Houses Prices')
        for column_to_keep in columns_to_keep:
            visualize_scatter_plot(houses_prices_df, column_to_keep, y_column)
        # for build_options in (1,);
        #     nn = build_full_connected_nn(*build_options)


if __name__ == '__main__':
    main()
