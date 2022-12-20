# libraries
import pandas as pd
import numpy as np


def get_train_set(c, df):
    '''
    Filters dataframe based on train dates.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Historical Time Series.

    Returns
    -------
    df_train : pandas dataframe
        Train dataframe.

    '''

    if c.use_test_set:
        df_train = df[df[c.date_column] <= c.end_history_date]
    else:
        df_train = df.copy()

    return df_train


def get_lifecycle(c, df):
    '''
    Classifies a Time Series in one of the following categories:
        - less_1_period_history: Time Series has less than 1 full period
                                 of history.
        - obsolete: Time Series has null values for a considerable amount in
                    its recent history.
        - consolidated: None of the above. Time Series is stable in time.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.

    Returns
    -------
    lifecycle : pandas dataframe
        Dataframe with Time Series, and first and last day of the Time Series
        existance.

    '''

    min_dates = (
        df[df[c.target_column] != 0].groupby(c.forecast_group_level)
        [c.date_column].min().reset_index(name='min_date')
    )
    max_dates = (
        df[df[c.target_column] != 0].groupby(c.forecast_group_level)
        [c.date_column].max().reset_index(name='max_date')
    )
    lifecycle = pd.merge(
        left=min_dates, right=max_dates, on=c.forecast_group_level, how='left'
    )

    lifecycle.loc[
        lifecycle['min_date'] > c.end_history_date - pd.DateOffset(c.period),
        'lifecycle'
    ] = 'less_1_period_history'

    lifecycle.loc[
        lifecycle['max_date']
        < c.end_history_date - pd.DateOffset(int(c.period*0.85)),
        'lifecycle'
    ] = 'obsolete'

    lifecycle.loc[
        lifecycle['lifecycle'].isnull(),
        'lifecycle'
    ] = 'consolidated'

    return lifecycle


def filter_ts_based_on_lifecycle(c, df, lifecycle):
    '''
    Remove initial period where the Time Series did not exist yet.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.
    lifecycle : pandas dataframe
        Time Series lifecycle info.

    Returns
    -------
    df : pandas dataframe
        Dataframe with Time Series.

    '''

    df = pd.merge(
        left=df, right=lifecycle, on=c.forecast_group_level, how='left'
    )
    df = df[df[c.date_column] >= df['min_date']]
    df = df.drop(['min_date', 'max_date'], axis=1)

    return df


def add_missing_dates(c, df, start_date, end_date):
    '''
    Fills missing Time Series points with 0s.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe to add missing dates to.
    start_date : pandas timestamp
        First date to start completition.
    end_date : pandas timestamp
        Last date to end completition.

    Returns
    -------
    df : pandas dataframe
        Dataframe with the missing dates added.

    '''

    history_dates = pd.date_range(start=start_date, end=end_date)
    unique_ts = (
        df.groupby(c.forecast_group_level).size().reset_index()
        [c.forecast_group_level]
    )
    complete_dates = pd.DataFrame(
        data=np.repeat(unique_ts.values, len(history_dates), axis=0),
        columns=unique_ts.columns
    )
    complete_dates[c.date_column] = list(history_dates) * len(unique_ts)

    df = pd.merge(
        left=complete_dates, right=df,
        on=c.forecast_group_level + [c.date_column], how='left'
    )
    df = df.fillna(0)

    return df


def prepare_forecast_inputs(c, df):
    '''
    Creates a list of tuples, each tuple containing the necessary arguments to
    be passed to the compute_forecast() function to generate a forecast for
    1 Time Series.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe containing all Time Series.

    Returns
    -------
    inputs : list of tupples
        List where each element (tupple) contains the necessary inputs to
        forecast 1 Time Series.

    '''

    inputs = []

    # the order matters for the next steps!
    df = df.sort_values(
        by=c.forecast_group_level + [c.date_column], ascending=True
    )

    df_splited_by_ts = [d for _, d in df.groupby(c.forecast_group_level)]

    for ts in df_splited_by_ts:
        inputs.append(tuple([c, ts]))

    return inputs
