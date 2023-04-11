# libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error


def get_rmse_test_and_plots(c, df, lifecycle, forecast, plot_results):
    '''
    Calculates RMSE for the test set and plots the forecast.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe with Time Series.
    forecast : pandas dataframe
        Forecast dataframe.
    plot_results : bool
        Indicates if we want plots to be ploted or not.

    Returns
    -------
    rmse_test : pandas dataframe
        RMSE for each Time Series.

    '''

    # exclude Time Series that exist in the test set but not in the train set
    df = pd.merge(
        left=lifecycle[c.forecast_group_level], right=df,
        on=c.forecast_group_level, how='left'
    )

    # split each Time Series in individual dataframes
    df_train = (
        df[df[c.date_column] <= c.end_history_date]
        .sort_values(by=c.forecast_group_level, ascending=True)
    )
    df_test = (
        df[df[c.date_column] >= c.start_predict_date]
        .sort_values(by=c.forecast_group_level, ascending=True)
    )
    forecast = forecast.sort_values(by=c.forecast_group_level, ascending=True)

    df_train_splited = [d for _, d in df_train.groupby(c.forecast_group_level)]
    df_test_splited = [d for _, d in df_test.groupby(c.forecast_group_level)]
    forecast_splited = [d for _, d in forecast.groupby(c.forecast_group_level)]

    if c.use_test_set:
        # initialize rmse_test dataframe
        rmse_test = pd.DataFrame(columns=c.forecast_group_level + ['rmse'])

        for ts_train, ts_test, ts_forecast in zip(
            df_train_splited, df_test_splited, forecast_splited
        ):
            # calculate RMSE
            rmse_ts = ts_train[c.forecast_group_level].head(1)
            rmse_ts['rmse'] = round(
                mean_squared_error(
                    list(ts_test[c.target_column]),
                    list(ts_forecast['forecast'])
                )**0.5,
                2
            )
            rmse_test = pd.concat([rmse_test, rmse_ts], axis=0)

            # plot results
            if plot_results:
                plot_title = get_plot_title(
                    c=c, ts_id=list(ts_train[c.forecast_group_level].iloc[0]),
                    rmse_ts=rmse_ts
                )
                plot_ts(
                    c=c, ts_train=ts_train, ts_test=ts_test,
                    ts_forecast=ts_forecast, plot_title=plot_title
                )


    else:
        # calculate RMSE
        rmse_test = None

        # plot results
        if plot_results:
            for ts_train, ts_forecast in zip(
                df_train_splited, forecast_splited
            ):
                plot_title = get_plot_title(
                    c=c, ts_id=list(ts_train[c.forecast_group_level].iloc[0])
                )
                plot_ts(
                    c=c, ts_train=ts_train, ts_test=None,
                    ts_forecast=ts_forecast, plot_title=plot_title
                )

    return rmse_test


def get_plot_title(c, ts_id, rmse_ts=None):
    '''
    Creats the title of the plot based on the ts_id that is beeing ploted.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    ts_id : list
        List containing each parameter that identifies the ts_id.
    rmse_ts: pandas dataframe
        Metrics of the Time Series.

    Returns
    -------
    plot_title : string
        Plot title.

    '''

    plot_title = ''

    for i, j in zip(c.forecast_group_level, ts_id):
        aux = i + ': ' + j + ' | '
        plot_title = plot_title + aux

    plot_title = plot_title[:-3]

    if rmse_ts is not None:
        plot_title = f"""{plot_title} | RMSE: {rmse_ts['rmse'].iloc[0]}"""

    return plot_title


def plot_ts(c, ts_train, ts_test, ts_forecast, plot_title):
    '''
    Plots the historical, expected and forecsted values of the Time Series.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    ts_train : pandas dataframe
        Time Series train dataframe (historic values).
    ts_test : pandas dataframe
        Time Series test dataframe (expected values).
    ts_forecast : pandas dataframe
        Time Series forecast dataframe.
    plot_title : string
        Plot title.

    Returns
    -------
    None.

    '''

    font = {'size': 22}
    plt.rc('font', **font)
    figure(figsize=(20, 10), dpi=80)
    plt.plot(
        c.history_dates, list(ts_train[c.target_column]),
        label='history'
    )
    if c.use_test_set:
        plt.plot(
            c.predict_dates, list(ts_test[c.target_column]),
            label='expected_output'
        )
    plt.plot(
        c.predict_dates, list(ts_forecast['forecast']),
        label='forecast'
    )
    plt.legend(loc='upper left')
    plt.title(plot_title)
    plt.show()
