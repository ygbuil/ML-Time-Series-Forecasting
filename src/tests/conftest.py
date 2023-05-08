# libraries
import pytest
import pandas as pd
import numpy as np
from random import randrange

# local libraries
import objects.main_modules as m
from constants.constants import c


@pytest.fixture
def test_df():
    df = m.read_inputs(c=c, file_path='data/products_sales.csv')
    test_df = df[c.forecast_group_level].drop_duplicates().head(3)
    test_df = pd.merge(
        left=test_df, right=df, on=c.forecast_group_level, how='left'
    )

    return test_df


@pytest.fixture
def df_train():
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
