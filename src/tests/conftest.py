# libraries
import pytest
import pandas as pd
from random import randrange

# local libraries
from constants.constants import c


@pytest.fixture
def df():
    '''
    Create original input dataframe with 3 skus / time series.

    Returns
    -------
    df : pandas dataframe
        Input dataframe.

    '''

    # define number of skus of the df
    n_skus = 3

    # create dates
    if c.use_test_set:
        dates = pd.date_range(
            start=c.start_history_date, end=c.end_predict_date
        )
    else:
        dates = pd.date_range(
            start=c.start_history_date, end=c.end_history_date
        )

    # create names for the forecast_group_level categories
    forecast_group_level_labels = []
    for col in c.forecast_group_level:
        temp = []
        for i in range(n_skus):
            temp += [f'{col}_test_{i}']*len(dates)
        forecast_group_level_labels.append(temp)

    # generate random time series values
    ts = [randrange(100) for x in range(len(dates)*n_skus)]

    df = pd.DataFrame(data=dict(zip(
        c.forecast_group_level + [c.date_column, c.target_column],
        [*forecast_group_level_labels, list(dates)*n_skus, ts])
    ))

    return df


@pytest.fixture
def df_train():
    '''
    Create train dataframe for 1 sku, ready to be used for prediction (except
    for the attributes, which are missing)

    Returns
    -------
    df_train : pandas dataframe
        Dataframe for 1 sku.

    '''

    history_dates = pd.date_range(
        start=c.start_history_date, end=c.end_history_date
    )

    forecast_group_level_labels = [
        [x + '_test']*len(history_dates) for x in c.forecast_group_level
    ]

    ts = [randrange(100) for x in range(len(history_dates))]

    df_train = pd.DataFrame(data=dict(zip(
        c.forecast_group_level + [c.date_column, c.target_column],
        [*forecast_group_level_labels, history_dates, ts])
    ))

    return df_train
