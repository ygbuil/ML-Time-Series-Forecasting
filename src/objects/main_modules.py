# libraries
import sys
import pandas as pd

# local libraries
import objects.general as gen
import objects.features as ft
import objects.forecasting as fcst
import objects.results as res


def define_sys_path(c):
    '''
    Set sys path to the root path. This is very important for multiprocessing
    library to work properly when parallelising the forecast.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.

    Returns
    -------
    None.

    '''

    sys.path.append(c.root_path) if c.root_path not in sys.path else None


def read_inputs(c, file_path):
    '''
    Read input data.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    file_path : string
        Time Series csv file path.

    Returns
    -------
    df : pandas dataframe
        Dataframe with Time Series.

    '''

    df = pd.read_csv(file_path)
    df[c.date_column] = pd.to_datetime(df[c.date_column])
    df = df.sort_values(
        by=c.forecast_group_level + [c.date_column], ascending=True
    )

    return df


def preprocessing(c, df):
    '''
    Apply preprocessing to the initial dataframe: add missing values, add model
    features...

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.

    Returns
    -------
    df : pandas dataframe
        Dataframe with Time Series preprocessed.
    lifecycle : pandas dataframe
        Time Series lifecycle info.
    inputs : tupple
        Inputs that will be used for forecasting.

    '''

    df = gen.add_missing_dates(
        c=c, df=df, start_date=c.start_history_date,
        end_date=c.end_predict_date if c.use_test_set else c.end_history_date
    )

    df_train = gen.get_train_set(c=c, df=df)

    lifecycle = gen.get_lifecycle(c=c, df=df_train)

    df_train = ft.add_features(c=c, df=df_train)

    df_train = gen.filter_ts_based_on_lifecycle(
        c=c, df=df_train, lifecycle=lifecycle
    )

    inputs = gen.prepare_forecast_inputs(c=c, df=df_train)

    return df, lifecycle, inputs


def forecasting(c, inputs):
    '''
    Generate forecast.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    inputs : tupple
        Inputs that will be used for forecasting.

    Returns
    -------
    forecast : pandas dataframe
        Dataframe with forecasted Time Series.

    '''

    forecast = fcst.compute_forecast(
        c=c, inputs=inputs, parallel_forecast=c.parallel_forecast
    )

    return forecast


def results(c, df, forecast, lifecycle):
    '''
    Calculates RMSE for the test set and plots the forecast.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with historical Time Series.
    forecast : pandas dataframe
        Dataframe with forecasted Time Series.
    lifecycle : pandas dataframe
        Time Series lifecycle info.

    Returns
    -------
    rmse_test : pandas dataframe
        RMSE for each Time Series.

    '''

    rmse_test = res.get_rmse_test_and_plots(
        c=c, df=df, forecast=forecast, lifecycle=lifecycle,
        plot_results=c.plot_results
    )

    return rmse_test
