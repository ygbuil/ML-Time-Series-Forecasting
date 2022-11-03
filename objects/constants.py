# libraries
import pandas as pd


class Constants:
    def __init__(
        self, forecast_group_level, date_column, target_column, use_test_set,
        period, start_history_date, end_history_date, start_predict_date,
        end_predict_date, missing_dates, lag_features, list_of_lags,
        statistical_features, stats_to_calculate, number_of_periods,
        shift_of_periods, slope_features, slope_past_periods, time_features,
        time_features_encoding_type, use_cross_validation, cv_n_splits,
        xgb_hyperparams
    ):
        '''
        Declares all the constants that will be using during the entire code.

        Parameters
        ----------
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

        Returns
        -------
        None.

        '''

        self.forecast_group_level = forecast_group_level
        self.date_column = date_column
        self.target_column = target_column
        self.use_test_set = use_test_set
        self.period = period
        self.start_history_date = start_history_date
        self.end_history_date = end_history_date
        self.start_predict_date = start_predict_date
        self.end_predict_date = end_predict_date
        self.history_dates = pd.date_range(
            start=self.start_history_date, end=self.end_history_date, freq='D'
        )
        self.predict_dates = pd.date_range(
            start=self.start_predict_date, end=self.end_predict_date, freq='D'
        )
        self.missing_dates = missing_dates
        self.lag_features = lag_features
        self.list_of_lags = list_of_lags
        self.statistical_features = statistical_features
        self.stats_to_calculate = stats_to_calculate
        self.number_of_periods = number_of_periods
        self.shift_of_periods = shift_of_periods
        self.slope_features = slope_features
        self.slope_past_periods = slope_past_periods
        self.time_features = time_features
        self.time_features_encoding_type = time_features_encoding_type
        self.use_cross_validation = use_cross_validation
        self.cv_n_splits = cv_n_splits
        self.xgb_hyperparams = xgb_hyperparams

    def run_checks(self):
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
