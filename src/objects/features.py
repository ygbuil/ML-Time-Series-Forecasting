# libraries
import numpy as np
import pandas as pd
from scipy.stats import linregress


def add_features(c, df):
    '''
    Adds various features to the input dataframe.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.

    Returns
    -------
    df : pandas dataframe
        Dataframe with features added.

    '''

    if c.lag_features:
        df = add_lags(c=c, df=df)
    if c.statistical_features:
        df = add_statistical_features(c=c, df=df)
    if c.slope_features:
        df = add_slope(c=c, df=df, slope_past_periods=c.slope_past_periods)
    if c.time_features:
        df = add_time_features(c=c, df=df)

    return df


def add_lags(c, df):
    '''
    Adds past Time Series values in column format.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.

    Returns
    -------
    df : pandas dataframe
        Dataframe with the lags added.

    '''

    for i in c.list_of_lags:
        lag_column = (
            df.groupby(c.forecast_group_level)[c.target_column].shift(i)
            .rename(f'lag_{i}')
        )

        df = pd.concat([df, lag_column], axis=1)

    return df


def add_statistical_features(c, df):
    '''
    Adds rolling stats (mean, sum, std).

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.

    Returns
    -------
    df : pandas dataframe
        Dataframe with the stats added.

    '''

    for stat, period, shift in zip(
        c.stats_to_calculate, c.number_of_periods, c.shift_of_periods
    ):
        new_column_name = f'{stat}_last_{period}_periods_{shift}_shift'

        if stat == 'mean':
            df[new_column_name] = (
                df.groupby(c.forecast_group_level)[c.target_column]
                .shift(shift).rolling(period).mean()
            )
        elif stat == 'sum':
            df[new_column_name] = (
                df.groupby(c.forecast_group_level)[c.target_column]
                .shift(shift).rolling(period).sum()
            )
        elif stat == 'std':
            df[new_column_name] = (
                df.groupby(c.forecast_group_level)[c.target_column]
                .shift(shift).rolling(period).std()
            )

    return df


def add_slope(c, df, slope_past_periods):
    '''
    Calculates the slope of a liniear regression fitted to the last n Time
    Series periods.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.
    slope_past_periods : int
        Number of past periods to use to calculate the slope of the linear
        regression.

    Returns
    -------
    df : pandas dataframe
        Dataframe with the slopes added.

    '''

    df['slope'] = (
        df.groupby(c.forecast_group_level)[c.target_column].shift(1)
        .rolling(c.slope_past_periods).apply(
            lambda x: linregress(range(len(x)), x).slope, raw=True
        )
    )

    return df


def add_time_features(c, df):
    '''
    Adds various time features (day_of_week, month, season...) with 3 different
    types of encoding options:
        - one_hot_encoding: One hot encoding.
        - sin_cos_encoding: Each value is represented by the projection of a
                            vector on the x and y axis.
        - target_encoding: Expanding mean of target column for each time
                           feature.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.

    Returns
    -------
    df : pandas dataframe
        Dataframe with the time features added.

    '''

    season_dic = {
        1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4,
        12: 1
    }
    trimester_dic = {
        1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4,
        12: 4
    }
    time_features = [
        'day_of_week', 'week_of_year', 'month', 'season', 'trimester'
    ]

    df['day_of_week'] = df[c.date_column].dt.dayofweek
    df['week_of_year'] = df[c.date_column].dt.isocalendar().week
    df['month'] = df[c.date_column].dt.month
    df['season'] = df[c.date_column].dt.month.map(season_dic)
    df['trimester'] = df[c.date_column].dt.month.map(trimester_dic)

    if c.time_features_encoding_type == 'one_hot_encoding':
        df = pd.get_dummies(data=df, columns=time_features)

    elif c.time_features_encoding_type == 'sin_cos_encoding':
        df['day_of_week_sin'] = np.sin((df['day_of_week']+1)/7 * 2*np.pi)
        df['day_of_week_cos'] = np.cos((df['day_of_week']+1)/7 * 2*np.pi)
        df['week_of_year_sin'] = np.sin(df['week_of_year']/53 * 2*np.pi)
        df['week_of_year_cos'] = np.cos(df['week_of_year']/53 * 2*np.pi)
        df['month_sin'] = np.sin(df['month']/12 * 2*np.pi)
        df['month_cos'] = np.cos(df['month']/12 * 2*np.pi)
        df['season_sin'] = np.sin(df['season']/4 * 2*np.pi)
        df['season_cos'] = np.cos(df['season']/4 * 2*np.pi)
        df['trimester_sin'] = np.sin(df['trimester']/4 * 2*np.pi)
        df['trimester_cos'] = np.cos(df['trimester']/4 * 2*np.pi)

        df = df.drop(time_features, axis=1)

    elif c.time_features_encoding_type == 'target_encoding':
        for time_feature in time_features:
            df[f'target_encoded_{time_feature}'] = (
                df.groupby(c.forecast_group_level + [time_feature])
                [c.target_column]
                .apply(lambda x: x.expanding().mean().shift(1))
            )

        df = df.drop(time_features, axis=1)

    return df


def add_lags_one_row(c, df, history):
    '''
    Adds past Time Series values in column format.
    Same functionality as add_lags() but for one row only instead of all
    dataframe.
    The purpose of this function is to save computation time when only one new
    row needs to be created.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Row to add lags to.
    history : list of floats
        List with historic Time Series.

    Returns
    -------
    df : pandas dataframe
        One row dataframe with the lags added.

    '''

    for lag_number in c.list_of_lags:
        df[f'lag_{lag_number}'] = history[-lag_number]

    return df


def add_statistical_features_one_row(c, df, history):
    '''
    Adds rolling stats (mean, sum, std).
    Same functionality as add_statistical_features() but for one row only
    instead of all dataframe.
    The purpose of this function is to save computation time when only one new
    row needs to be created.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Row to add stats features to.
    history : list of floats
        List with historic Time Series.

    Returns
    -------
    df : pandas dataframe
        One row dataframe with the stats added.

    '''

    history = list(reversed(history))

    for stat, period, shift in zip(
        c.stats_to_calculate, c.number_of_periods, c.shift_of_periods
    ):
        new_column_name = f'{stat}_last_{period}_periods_{shift}_shift'

        if stat == 'mean':
            df[new_column_name] = (
                np.array(history[shift-1:shift+period-1]).mean()
            )
        elif stat == 'sum':
            df[new_column_name] = (
                np.array(history[shift-1:shift+period-1]).sum()
            )
        elif stat == 'std':
            df[new_column_name] = (
                np.array(history[shift-1:shift+period-1]).std()
            )

    return df


def add_slope_one_row(c, df, history):
    '''
    Calculates the slope of a liniear regression fitted to the last n Time
    Series periods.
    Same functionality as add_slope() but for one row only instead of all
    dataframe.
    The purpose of this function is to save computation time when only one new
    row needs to be created.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Row to add slope to.
    history : list of floats
        Past history.

    Returns
    -------
    df : pandas dataframe
        One row dataframe with the slope added.

    '''

    history = history[-c.slope_past_periods:]
    df['slope'] = linregress(range(len(history)), history).slope

    return df


def target_encode_predict(c, df, new_row):
    '''
    Target encoding feature generation for the iterative forecast prediction.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Iterative dataframe.
    new_row : pandas dataframe
        Row to add target encoding to.

    Returns
    -------
    new_row : pandas dataframe
        Row with target encoded time features added.

    '''

    aux = pd.concat([df, new_row], axis=0).reset_index(drop=True)
    new_row = add_time_features(c=c, df=aux).tail(1)

    return new_row
