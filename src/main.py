# local libraries
import objects.main_modules as m
from constants.constants import c


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

    # define sys path as root path
    m.define_sys_path(c=c)

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
