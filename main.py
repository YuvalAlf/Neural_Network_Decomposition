import matplotlib.pyplot as plt
import numpy as np
from utils.dataframes_utils import read_dataset, filter_dataset
from utils.visualization_utils import visualize_histogram




def main():
    sale_price = 'SalePrice'
    columns_to_keep = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF',
                       'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'YrSold', 'SalePrice']
    houses_prices_df = read_dataset('datasets/houses_prices.csv')
    houses_prices_df = filter_dataset(houses_prices_df, columns_to_keep)
    visualize_histogram(houses_prices_df[sale_price].values, 'Price', 'Houses Prices')
    visualize_histogram(np.log(houses_prices_df[sale_price].values), 'Log[Price]', 'Houses Prices [Log]')


if __name__ == '__main__':
    main()
