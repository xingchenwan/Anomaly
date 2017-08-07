import pandas as pd
import numpy as np
import logging
import scipy
import statsmodels.api as sm
from settings import *
from Detectors.Prototype import *


class ProbabilisticEWMA(Prototype):
    def __init__(self, input_series, window_size=None,
                 trend_period=None, sigma=None, beta=None):
        self.label = "PEWMA"
        window_size = window_size if window_size else PEWMA_WINDOW_SIZE
        super(ProbabilisticEWMA).__init__(input_series, window_size)
        self.trend_period = trend_period if trend_period else PEWMA_TREND_PERIOD
        self.sigma = sigma if sigma else PEWMA_SIGMA
        self.beta = beta if beta else PEWMA_BETA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return 0
        curr_pt = self.series.iloc[loc]
        mean = local_series.mean()
        std = local_series.std()
        alpha = 2 /(self.trend_period + 1)
        z = (curr_pt - mean) / std if std != 0. else np.nan
        p = 0 if std == 0 else 1 / (np.sqrt(2*np.pi)) * np.exp(-z**2 / 2)

        adjusted_alpha = (1 - self.beta*p) * alpha
        pred = local_series.ewm(alpha=adjusted_alpha).mean().iloc[-1]
        anom_score = (curr_pt - pred)/std
        anom_status = 0

        if abs(anom_score) > self.sigma:
            anom_status = 1 if anom_score > 0 else -1
        return anom_status

    def series_detect(self):
        self._series_detect()


class SimpleMovingAvg(Prototype):
    def __init__(self, input_series, window_size=None, trend_period=None, sigma=None):
        self.label = "SMA"
        window_size = window_size if window_size else SMA_WINDOW_SIZE
        super(SimpleMovingAvg).__init__(input_series, window_size)
        self.trend_period = trend_period if trend_period else SMA_TREND_PERIOD
        self.sigma = sigma if sigma else SMA_SIGMA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return 0
        curr_pt = self.series.iloc[loc]
        local_mean = local_series.mean()
        std = local_series.std()
        anom_score = (curr_pt - local_mean)/std
        anom_status = 0

        if abs(anom_score) > self.sigma:
            anom_status = 1 if anom_score > 0 else -1
        return anom_status

    def series_detect(self):
        self._series_detect()


class KS(Prototype):
    def __init__(self, input_series, window_size=None, probe_size=None, alpha=None):
        self.label = "KS"
        window_size = window_size if window_size else KS_WINDOW_SIZE
        super(KS).__init__(input_series, window_size)
        if window_size < 20:
            raise Exception("Choose longer Window Size for Reliable"
                            "Kolmogorov-Smirnov test result!")
        if probe_size > window_size / 2:
            raise Exception("Reduce probe size or increase window size!")
        self.probe_size = probe_size if probe_size else KS_PROBE_SIZE
        self.alpha = alpha if alpha else KS_ALPHA

    def point_detect(self, loc):
        reference_series = self._get_local_series(self.series, self.window_size, loc-self.probe_size)
        adf = sm.tsa.stattools.adfuller(reference_series, 10)
        attempt_ct = 0
        while adf[1] < 0.05:
            reference_series = self._stationarise_series(reference_series, mode='detrend')
            adf = sm.tsa.stattools.adfuller(reference_series, 10)
            attempt_ct += 1
            if attempt_ct >= 3:
                logging.error("Unable to remove trend in the series. KS test aborted")
                return 0
        probe_series = self._get_local_series(self.series, self.window_size, loc)
        if reference_series is None or probe_series is None:
            return 0
        ks_d, ks_p_val = scipy.stats.ks_2samp(reference_series, probe_series)
        if ks_p_val > self.alpha and ks_d > 0.5:
            return 1

    @staticmethod
    def _stationarise_series(series, mode):
        if mode == 'difference':
            res = series.diff()
            res.dropna()
        elif mode == 'detrend':
            res = series.ewm(span=3).mean()
            res = res.dropna()
        else:
            logging.error("Unknown keyword for stationarising")
            res = series
        return res

    def series_detect(self):
        self._series_detect()


class Grubbs(Prototype):
    def __init__(self, input_series, window_size=None):
        self.label = "GRUBBS"
        window_size = window_size if window_size else GRUBBS_WINDOW_SIZE
        super(Grubbs).__init__(input_series, window_size)

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return 0
        local_mean = local_series.mean()
        std = local_series.std()
        curr_pt = self.series.iloc[loc]
        z = (curr_pt - local_mean)/std

        local_len = len(local_series)
        threshold = (scipy.stat.t.isf(.05 / (2*local_len), local_len-2)) ** 2
        grubbs = ((local_len - 1)/np.sqrt(local_len) * np.sqrt(threshold / (local_len - 2 + threshold)))

        if abs(z) > abs(grubbs):
            return 1
        return 0

    def series_detect(self):
        self._series_detect()


class Derivative(Prototype):
    def __init__(self, input_series, window_size=None, sigma=None):
        self.label = "DERIV"
        window_size = window_size if window_size else DERIV_WINDOW_SIZE
        super(Derivative).__init__(input_series, window_size)
        self.sigma = sigma if sigma else DERIV_SIGMA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return 0
        local_series_diff = local_series.diff()
        local_series_diff[0] = 0.
        curr_pt = self.series.iloc[loc]
        curr_pt_diff = curr_pt - local_series[-1]

        z = abs((curr_pt_diff - local_series_diff.mean()) / curr_pt_diff.std())
        if z > self.sigma:
            return 1
        return 0

    def series_detect(self):
        self._series_detect()
