# libraries
import pandas as pd

# local libraries
from src.objects.constants import Constants


# constants definitions
c = Constants(
    forecast_group_level=['product', 'store'],
    date_column='date',
    target_column='sales',
    use_test_set=True,
    period=365,
    start_history_date=pd.Timestamp('2019-01-01'),
    end_history_date=pd.Timestamp('2020-12-31'),
    start_predict_date=pd.Timestamp('2021-01-01'),
    end_predict_date=pd.Timestamp('2021-12-31'),
    missing_dates=True,
    lag_features=True,
    list_of_lags=[7, 364, 365, 366],
    statistical_features=True,
    stats_to_calculate=['mean'],
    number_of_periods=[90],
    shift_of_periods=[1],
    slope_features=True,
    slope_past_periods=30,
    time_features=True,
    time_features_encoding_type='one_hot_encoding',
    use_cross_validation=False,
    cv_n_splits=4,
    xgb_hyperparams={
        'learning_rate': [0.01, 0.1],
        'gamma': [0, 0.25, 1],
        'max_depth': [3, 6],
        'min_child_weight': [1, 10.0],
        'subsample': [0.5, 1],
        'n_estimators': [30, 100],
        'reg_alpha': [0.1, 50]
    },
    parallel_forecast=True,
    plot_results=True
)

# check for possible invalid constants
c.run_checks()
