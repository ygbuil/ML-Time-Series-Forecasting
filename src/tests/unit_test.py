# libraries
import pandas as pd
from random import randrange

# local libraries
import objects.general as gen
import objects.features as ft
import objects.forecasting as fcst
from constants.constants import c


def test_add_missing_dates(test_df):
    # get unique skus
    unique_skus = len(test_df[c.forecast_group_level].drop_duplicates())

    # get complete dates
    history_dates = pd.date_range(
        start=c.start_history_date,
        end=c.end_predict_date if c.use_test_set else c.end_history_date
    )

    # remove random date
    test_df = test_df[
        test_df[c.date_column] != history_dates[randrange(len(history_dates))]
    ]

    # add missing dates
    test_df = gen.add_missing_dates(
        c=c, df=test_df, start_date=c.start_history_date,
        end_date=c.end_predict_date if c.use_test_set else c.end_history_date
    )

    assert len(test_df) == len(history_dates)*unique_skus


def test_xgb_forecast(df_train):
    df_train = ft.add_features(c=c, df=df_train)

    forecast = fcst.xgb_forecast(c, df_train)['forecast']

    predict_dates = pd.date_range(
        start=c.start_predict_date, end=c.end_predict_date
    )

    assert len(forecast) == len(predict_dates)
