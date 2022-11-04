# libraries
import os
import sys

# root path
path = 'C:\\Users\\llorenc.buil\\Documents\\ML-Time-Series-Forecasting\\'
os.chdir(path)
if path not in sys.path:
    sys.path.append(path)

# local libraries
from pipeline import modules
from constants.constants import c


def pipeline(c):
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
