# libraries
import pandas as pd
from random import randrange

# local libraries
import objects.general as gen
import objects.features as ft
import objects.forecasting as fcst
from constants.constants import c


def test_add_missing_dates(df, n_skus):
    '''
    Tests add_missing_dates() by removing a random date to the input dataframe
    and then filling the empty date.

    Parameters
    ----------
    df : pandas dataframe
        Input dataframe.
    n_skus : int
        Number of unique skus used for testing.

    Returns
    -------
    None.

    '''

    # get complete dates
    history_dates = pd.date_range(
        start=c.start_history_date,
        end=c.end_predict_date if c.use_test_set else c.end_history_date
    )

    # remove random date
    df = df[
        df[c.date_column] != history_dates[randrange(len(history_dates))]
    ]

    # add missing dates
    df = gen.add_missing_dates(
        c=c, df=df, start_date=c.start_history_date,
        end_date=c.end_predict_date if c.use_test_set else c.end_history_date
    )

    assert len(df) == len(history_dates)*n_skus


def test_get_lifecycle(df):
    '''
    Test get_lifecycle(). Checks that every sku has a lifecycle assigned.

    Parameters
    ----------
    df : pandas dataframe
        Input dataframe.

    Returns
    -------
    None.

    '''

    df_train = df[df[c.date_column] <= c.end_history_date]

    lifecycle = gen.get_lifecycle(c=c, df=df_train)

    assert not all(lifecycle['lifecycle'].isnull())


def test_xgb_forecast(df_train):
    '''
    Test XGBoost forecast generation. Checks that the forecast output is the
    same length as the expected prediction horizon.

    Parameters
    ----------
    df_train : pandas dataframe
        Train dataframe.

    Returns
    -------
    None.

    '''

    df_train = ft.add_features(c=c, df=df_train)

    forecast = fcst.xgb_forecast(c, df_train)['forecast']

    predict_dates = pd.date_range(
        start=c.start_predict_date, end=c.end_predict_date
    )

    assert len(forecast) == len(predict_dates)
