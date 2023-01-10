# libraries
import os
import sys

# root path
path = 'C:\\Users\\llorenc.buil\\Documents\\ML-Time-Series-Forecasting\\'
os.chdir(path)
if path not in sys.path:
    sys.path.append(path)

# local libraries
from src.pipeline import modules
from src.constants.constants import c


def pipeline(c):
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
    metrics : pandas dataframe
        Metrics dataframe contianing MAE and RMSE.

    '''

    # read inputs
    df = modules.read_inputs(c=c, file_path='data/example_dataset.csv')

    # preprocessing
    df, lifecycle, inputs = modules.preprocessing(c=c, df=df)

    # forecasting
    forecast = modules.forecasting(c=c, inputs=inputs)

    # results
    metrics = modules.results(
        c=c, df=df, forecast=forecast, lifecycle=lifecycle
    )

    return forecast, metrics


forecast, metrics = pipeline(c=c)
