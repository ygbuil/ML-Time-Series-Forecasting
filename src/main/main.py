# libraries
import os
import sys

# root path
path = 'C:\\Users\\llorenc.buil\\github\\ML-Time-Series-Forecasting'
os.chdir(path)
if path not in sys.path:
    sys.path.append(path)

# local libraries
from src.constants.constants import c
from src.objects import main_modules as m


def main(c):
    '''
    Main pipeline that runs the entire process and outputs the forecast.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.

    Returns
    -------
    forecast : pandas dataframe
        Forecast.
    rmse_validation : pandas dataframe
        RMSE for the validation set.
    rmse_test : pandas dataframe
        RMSE for the test set.

    '''

    # read inputs
    df = m.read_inputs(c=c, file_path='data/products_sales.csv')

    # preprocessing
    df, lifecycle, inputs = m.preprocessing(c=c, df=df)

    # forecasting
    forecast, rmse_validation = m.forecasting(c=c, inputs=inputs)

    # results
    rmse_test = m.results(c=c, df=df, forecast=forecast, lifecycle=lifecycle)

    return forecast, rmse_validation, rmse_test


if __name__ == '__main__':
    forecast, rmse_validation, rmse_test = main(c=c)
