# libraries
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Constants:
    '''
    Declares all the constants that will be using during the entire code.

    Parameters
    ----------
    root_path : str
        Root path of the project. The root path is the 'src' folder.
        E.g.: 'C:\\Users\\...\\ML-Time-Series-Forecasting\\src'
    forecast_group_level : list of strings
        Level at which the forecast will be calculated.
    date_column : string
        Date column name.
    target_column : string
        Target column name.
    use_test_set : bool
        Indicates if test set will be used to check model performance.
    period: int
        Periodicity of the Time Series (365 for daily data, 12 for monthly
        data...).
    start_history_date : pandas timestamp
        First day of history used for training.
    end_history_date : pandas timestamp
        Last day of history used for training.
    start_predict_date : pandas timestamp
        First date to predict.
    end_predict_date : pandas timestamp
        Last date to predict.
    missing_dates : bool
        Indicates if missing dates have to be added to the Time Series.
    lag_features : bool
        Indicates if lag features have to be added to be calculated for
        training.
    list_of_lags : list of ints
        Lags to be added as features.
    statistical_features : bool
        Indicates if statistical features have to be added to be calculated
        for training.
    stats_to_calculate : list of strings
        Stats to be calculated as features.
        Options: ['mean', 'sum', 'std'].
    number_of_periods : list of ints
        Periods backwards to calculate stats_to_calculate. E.g.: [7, 30].
    shift_of_periods : list of ints
        Shift to apply to number_of_periods. Must be >= 1.
    slope_features : bool
        Indicates if slope feature has to be added to be calculated for
        training.
    slope_past_periods : int
        Number of past periods to use to calculate the slope of the linear
        regression.
    time_features : bool
        Indicates if time features have to be added to be calculated for
        training.
    time_features_encoding_type : string
        Type of encoding to apply to time features. Options:
            - one_hot_encoding: One hot encoding.
            - sin_cos_encoding: Each value is represented by the projection
                                of a vector on the x and y axis.
            - target_encoding: Expanding mean of target column for each
                               time feature.
    use_cross_validation : bool
        False: train with default XGBoost hyper parameters.
        True: cross validate xgb_hyperparams.
    cv_n_splits : int
        Number of splits for TimeSeriesSplit cross validation.
    xgb_hyperparams : dictionary
        Declares hyperparmeters to try.
    parallel_forecast : bool
        True: Forecast each Time Series in parallel.
        False: Forecast each Time Series sequentialy.
    plot_results : bool
        True: Plot forecasted Time Series.
        False: Do not plot.

    Returns
    -------
    None.

    '''

    root_path: str
    forecast_group_level: list
    date_column: str
    target_column: str
    use_test_set: bool
    period: int
    start_history_date: pd.Timestamp
    end_history_date: pd.Timestamp
    start_predict_date: pd.Timestamp
    end_predict_date: pd.Timestamp
    history_dates: pd.DatetimeIndex = field(init=False, default=None)
    predict_dates: pd.DatetimeIndex = field(init=False, default=None)
    missing_dates: bool
    lag_features: bool
    list_of_lags: list
    statistical_features: bool
    stats_to_calculate: list
    number_of_periods: list
    shift_of_periods: list
    slope_features: bool
    slope_past_periods: int
    time_features: bool
    time_features_encoding_type: str
    use_cross_validation: bool
    cv_n_splits: int
    xgb_hyperparams: dict
    parallel_forecast: bool
    plot_results: bool

    def __post_init__(self):
        self.history_dates = pd.date_range(
            start=self.start_history_date, end=self.end_history_date, freq='D'
        )
        self.predict_dates = pd.date_range(
            start=self.start_predict_date, end=self.end_predict_date, freq='D'
        )

    def run_checks(self):
        '''
        Check for incorrect input constants

        Raises
        ------
        Exception
            Exception message indicating the error.

        Returns
        -------
        None.

        '''

        for stat in self.stats_to_calculate:
            if stat not in ['mean', 'sum', 'std']:
                raise Exception(
                    """statistical_features can only take values:
                       'mean'
                       'sum'
                       'std'"""
                )

        for shift in self.shift_of_periods:
            if shift < 1:
                raise Exception(
                    """Shift_of_periods must be >= 1,
                    otherwise there is data leakage."""
                )

        if self.time_features_encoding_type not in [
            'one_hot_encoding', 'sin_cos_encoding', 'target_encoding'
        ]:
            raise Exception(
                """time_features_encoding_type can only take values:
                   'one_hot_encoding'
                   'sin_cos_encoding'
                   'target_encoding'"""
            )

        if not len(self.stats_to_calculate) == len(self.number_of_periods) \
            == len(self.shift_of_periods):
            raise Exception(
                """stats_to_calculate, number_of_periods and shift_of_periods
                   must be of the same length."""
            )
