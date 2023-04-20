# libraries
import os
import time
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# local libraries
import objects.features as ft


def compute_forecast(c, inputs, parallel_forecast=True):
    '''
    Iterates over every Time Series in parallel (or sequentially), training a
    model and making a prediction for each Time Series.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    inputs : list of tupples
        List of tupples where each tupple contains the inputs necessary for
        compute_forecast_single_ts().
    parallel_forecast : bool
        True = parallel forecast. False = sequential forecast.
        The default is True.

    Returns
    -------
    forecast : pandas dataframe
        Dataframe with entire forecast for every Time Series.
    rmse_validation: pandas dataframe
        Dataframe with the RMSE of the validation set for every Time Series.

    '''

    start = time.time()
    print('Start of forecast...')

    # parallel forecast
    if parallel_forecast:
        n_processes = calc_n_processes()
        chunksize = calc_chunksize(
            n_processes=n_processes, len_iterable=len(inputs)
        )

        with multiprocessing.Pool(processes=n_processes) as pool:
            forecast_results = list(
                tqdm(
                    pool.imap_unordered(
                        func=unpack_forecast_inputs, iterable=inputs,
                        chunksize=chunksize
                    ),
                    total=len(inputs)
                )
            )

    # sequential forecast
    else:
        forecast_results = [unpack_forecast_inputs(ts) for ts in inputs]

    # format results
    forecasts = [result['forecast'] for result in forecast_results]
    forecast = (
        pd.concat(forecasts).sort_values(c.forecast_group_level)
        .reset_index(drop=True)
    )

    if c.use_cross_validation:
        rmse_validations = [
            result['rmse_validation'] for result in forecast_results
        ]
        rmse_validation = (
            pd.concat(rmse_validations).sort_values(c.forecast_group_level)
            .reset_index(drop=True)
        )
    else:
        rmse_validation = None

    print('End of forecast!')
    print('Forecast duration:', round(time.time() - start, 2), 'sec')

    return forecast, rmse_validation


def unpack_forecast_inputs(inputs):
    '''
    Unpacks compute_forecast_single_ts() arguments.

    Parameters
    ----------
    inputs : list of tupples
        List of tupples where each tupple contains the inputs necessary for
        compute_forecast_single_ts().

    Returns
    -------
    pandas dataframe
        Output of compute_forecast_single_ts().

    '''

    return compute_forecast_single_ts(*inputs)


def compute_forecast_single_ts(c, df_train):
    '''
    Train and predict for 1 Time Series.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df_train : pandas dataframe
        Final dataframe to be used for training.

    Returns
    -------
    forecast_results : dict
        Dictionary containing the forecast and the RMSE of the validation set.

    '''

    if df_train['lifecycle'].iloc[0] == 'obsolete':
        df_train = df_train.drop('lifecycle', axis=1)
        forecast = obsolete_forecast(c=c, df_train=df_train)
        forecast_results = {'forecast': forecast, 'rmse_validation': None}

    elif df_train['lifecycle'].iloc[0] == 'less_1_period_history':
        df_train = df_train.drop('lifecycle', axis=1)
        forecast = less_1_period_forecast(c=c, df_train=df_train)
        forecast_results = {'forecast': forecast, 'rmse_validation': None}

    elif df_train['lifecycle'].iloc[0] == 'consolidated':
        df_train = df_train.drop('lifecycle', axis=1)
        forecast_results = xgb_forecast(c=c, df_train=df_train)

    return forecast_results


def obsolete_forecast(c, df_train):
    '''
    Returns 0 forecast for Time Series that have values 0 for a long period of
    time.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df_train : pandas dataframe
        Train dataframe.

    Returns
    -------
    forecast : pandas dataframe
        Dataframe containing the prediction.

    '''

    dates = pd.date_range(start=c.start_predict_date, end=c.end_predict_date)
    forecast = (
        pd.concat([df_train[c.forecast_group_level].head(1)] * len(dates))
        .reset_index(drop=True)
    )
    forecast[c.date_column] = dates
    forecast['forecast'] = [0] * len(dates)

    return forecast


def less_1_period_forecast(c, df_train):
    '''
    Calculates a Moving Average for Time Series with less than 1 period of
    history.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df_train : pandas dataframe
        Train dataframe.

    Returns
    -------
    forecast : pandas dataframe
        Dataframe containing the prediction.

    '''

    forecast_ts = moving_average(
        c=c, history=list(df_train[c.target_column]), rolling_window_size=90
    )
    dates = pd.date_range(start=c.start_predict_date, end=c.end_predict_date)

    forecast = (
        pd.concat([df_train[c.forecast_group_level].head(1)] * len(dates))
        .reset_index(drop=True)
    )
    forecast[c.date_column] = dates
    forecast['forecast'] = forecast_ts

    return forecast


def moving_average(c, history, rolling_window_size):
    '''
    Moving Average model.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    history : list
        Time Series to forecast.
    rolling_window_size : int
        Window size to be used to calculate the rolling mean.

    Returns
    -------
    forecast : list
        Forecasted Time Series.

    '''

    if len(history) == 0:
        forecast = [0] * len(c.predict_dates)

    else:
        # moving average
        for i in range(len(c.predict_dates)):
            history = history + [np.mean(history[-rolling_window_size:])]

        forecast = history[-len(c.predict_dates):]

    return forecast


def xgb_forecast(c, df_train):
    '''
    Calculates the forecast with the XGBoost model.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df_train : pandas dataframe
        Train dataframe.

    Returns
    -------
    forecast_results : dict
        Dictionary containing the forecast and the RMSE of the validation set.

    '''

    # X_train, y_train split
    X_train, y_train = (
        np.array(df_train.drop(
            c.forecast_group_level + [c.date_column, c.target_column], axis=1
        )),
        np.array(df_train[c.target_column])
    )

    if c.use_cross_validation:
        # declare estimatior, hyperparameters, cross validation method...
        model = GridSearchCV(
            estimator=XGBRegressor(objective='reg:squarederror', seed=0),
            param_grid=c.xgb_hyperparams,
            scoring='neg_root_mean_squared_error',
            verbose=10,
            cv=TimeSeriesSplit(n_splits=c.cv_n_splits)
        )

        # train
        model.fit(X_train, y_train)

        # get validation RMSE for the winning set of hyperparametrs, i.e. for
        # the final model (best model)
        rmse_validation_value = round(-model.best_score_, 2)
        rmse_validation = (
            df_train[c.forecast_group_level].head(1).reset_index(drop=True)
        )
        rmse_validation['rmse'] = rmse_validation_value

    else:
        # default XGBoost parameters
        model = XGBRegressor(objective='reg:squarederror', seed=0)

        # train
        model.fit(X_train, y_train)

        # rmse_validation set to None since no cross validation was applied
        rmse_validation = None

    # predict
    forecast = xgb_iterative_prediction(c=c, df_train=df_train, model=model)

    # final results
    forecast_results = {
        'forecast': forecast, 'rmse_validation': rmse_validation
    }

    return forecast_results


def xgb_iterative_prediction(c, df_train, model):
    '''
    Get prediction of future Time Series values. For every time step to be
    predicted, the function creates a row with all the necessary features,
    using information of the previous predictions.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df_train : pandas dataframe
        Train dataframe.
    model : ML model.
        ML model

    Returns
    -------
    df : pandas dataframe
        Dataframe containing the forecast.

    '''

    df_train_columns = pd.DataFrame(columns=df_train.columns)

    for next_date in c.predict_dates:
        # init new row
        new_row = pd.DataFrame(
            data=[list(df_train[c.forecast_group_level].iloc[0])
                  + [next_date, np.nan]],
            columns=c.forecast_group_level + [c.date_column, c.target_column]
        )

        # feature generation
        if c.lag_features:
            new_row = ft.add_lags_one_row(
                c=c, df=new_row, history=list(df_train[c.target_column])
            )
        if c.statistical_features:
            new_row = ft.add_statistical_features_one_row(
                c=c, df=new_row, history=list(df_train[c.target_column])
            )
        if c.slope_features:
            new_row = ft.add_slope_one_row(
                c=c, df=new_row, history=list(df_train[c.target_column])
            )
        if c.time_features:
            if c.time_features_encoding_type != 'target_encoding':
                new_row = ft.add_time_features(c=c, df=new_row)
            else:
                new_row = ft.target_encode_predict(
                    c=c, df=df_train, new_row=new_row
                )

        # add remaining columns and create predict_row
        new_row = (
            pd.concat([df_train_columns, new_row], axis=0, ignore_index=True)
            .fillna(0)
        )
        predict_row = new_row.drop(
            c.forecast_group_level + [c.date_column, c.target_column], axis=1
        )

        # make prediction
        new_row[c.target_column] = model.predict(np.array(predict_row))

        # add new_row to history
        df_train = pd.concat([df_train, new_row], axis=0, ignore_index=True)

    # take prediction period only
    forecast = df_train[df_train[c.date_column] >= c.start_predict_date]
    forecast = forecast[
        c.forecast_group_level + [c.date_column, c.target_column]
    ]
    forecast[c.date_column] = pd.to_datetime(forecast[c.date_column])
    forecast = forecast.rename(columns={c.target_column: 'forecast'})
    forecast.loc[forecast['forecast'] < 0, 'forecast'] = 0

    return forecast


def calc_n_processes():
    '''
    Calculates the optimal n_processes based on the CPUs available. For more
    than 60 n_processes the library is bugged so the value it is capped there.

    Returns
    -------
    n_processes : int
        Number of processes to spawn.

    '''

    n_processes = os.cpu_count() - 1

    if n_processes <= 60:
        return n_processes
    else:
        return 60


def calc_chunksize(n_processes, len_iterable, factor=4):
    '''
    Caclulated the optimal chunksize value.

    Parameters
    ----------
    n_processes : int
        Number of workers spawned.
    len_iterable : int
        Number of processes to calculate.
    factor : int
        Factor. The default is 4.

    Returns
    -------
    chunksize : int
        Chunksize.

    '''

    chunksize, extra = divmod(len_iterable, n_processes * factor)

    if extra:
        chunksize += 1

    return chunksize
