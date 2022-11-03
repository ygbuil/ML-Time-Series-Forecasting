# libraries
import pandas as pd

# local libraries
import objects.general as gen
import objects.features as ft
import objects.forecasting as fcst
import objects.results as res


def read_inputs(c, file_path):
    df = pd.read_csv(file_path)
    df[c.date_column] = pd.to_datetime(df[c.date_column])

    return df


def preprocessing(c, df):
    df = df.sort_values(
        by=c.forecast_group_level + [c.date_column], ascending=True
    )

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


def forecasting(inputs):
    forecast = fcst.compute_forecast(inputs=inputs, parallel_forecast=True)

    return forecast


def results(c, df, forecast, lifecycle):
    metrics = res.get_metrics_and_plots(
        c=c, df=df, forecast=forecast, lifecycle=lifecycle, plot=True
    )

    return metrics
