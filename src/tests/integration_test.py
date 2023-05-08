# libraries
import pandas as pd

# local libraries
import objects.main_modules as m
from constants.constants import c


def test_preprocessing(df):
    '''
    Tests the preprocessing module by checking that the sum of the target
    column sums the same value at the beginning vs end of preprocessing, as
    well as dimensions are preserved.

    Parameters
    ----------
    df : pandas dataframe
        Input dataframe.

    Returns
    -------
    None.

    '''

    # preprocessing
    df, lifecycle, inputs = m.preprocessing(c=c, df=df)
    df = df[df[c.date_column] <= c.end_history_date]

    # concat preprocessed data
    df_preprocessed = pd.concat([inputs[i][1] for i in range(len(inputs))])

    assert (
        sum(df[c.target_column]) == sum(df_preprocessed[c.target_column]) and
        len(df) == len(df_preprocessed)
    )


def test_preprocessing_and_forecast(df):
    '''
    Tests that the preprocessing and forecasting module interact properly
    between each other, making sure the final output is consistent with the
    input dimentions.

    Parameters
    ----------
    df : pandas dataframe
        Input dataframe.

    Returns
    -------
    None.

    '''

    # preprocessing
    df, lifecycle, inputs = m.preprocessing(c=c, df=df)

    # forecasting
    forecast, rmse_validation = m.forecasting(c=c, inputs=inputs)

    dates = pd.date_range(start=c.start_predict_date, end=c.end_predict_date)

    assert len(forecast) == len(list(dates)*3)
