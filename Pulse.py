import pandas as pd
import numpy as np
import logging
from scipy import stats
import statsmodels.api as sm
from settings import *
from Detectors.Prototype import *


class ProbabilisticEWMA(Prototype):
    def __init__(self, input_series, window_size=None,
                 trend_period=None, sigma=None, beta=None):
        self.label = "PEWMA"
        window_size = window_size if window_size else PEWMA_WINDOW_SIZE
        super(ProbabilisticEWMA,self).__init__(input_series, window_size)
        self.trend_period = trend_period if trend_period else PEWMA_TREND_PERIOD
        self.sigma = sigma if sigma else PEWMA_SIGMA
        self.beta = beta if beta else PEWMA_BETA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return np.nan
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
        super(SimpleMovingAvg, self).__init__(input_series, window_size)
        self.trend_period = trend_period if trend_period else SMA_TREND_PERIOD
        self.sigma = sigma if sigma else SMA_SIGMA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return np.nan
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
        self.probe_size = probe_size if probe_size else KS_PROBE_SIZE
        super(KS, self).__init__(input_series, window_size)
        if self.window_size < 20:
            raise Exception("Choose longer Window Size for Reliable"
                            "Kolmogorov-Smirnov test result!")
        if self.probe_size > self.window_size / 2:
            raise Exception("Reduce probe size or increase window size!")
        self.alpha = alpha if alpha else KS_ALPHA

    def point_detect(self, loc):
        reference_series = self._get_local_series(self.series, self.window_size, loc-self.probe_size)
        if reference_series is None: return np.nan
        adf = sm.tsa.stattools.adfuller(reference_series)
        attempt_ct = 0
        while adf[1] > 0.05:
            reference_series = self._stationarise_series(reference_series, mode='difference')
            adf = sm.tsa.stattools.adfuller(reference_series)
            attempt_ct += 1
            if attempt_ct >= 3:
                logging.error("Unable to remove trend in the series. KS test aborted")
                return np.nan
        probe_series = self._get_local_series(self.series, self.window_size, loc)
        if reference_series is None or probe_series is None:
            return np.nan
        ks_d, ks_p_val = stats.ks_2samp(reference_series, probe_series)
        if ks_p_val > self.alpha and ks_d > 0.5:
            return 1
        return 0

    @staticmethod
    def _stationarise_series(series, mode):
        if mode == 'difference':
            res = series.diff()
            res = res.dropna()
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
        super(Grubbs, self).__init__(input_series, window_size)

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return np.nan
        local_mean = local_series.mean()
        std = local_series.std()
        curr_pt = self.series.iloc[loc]
        z = (curr_pt - local_mean)/std

        local_len = len(local_series)
        threshold = (stats.t.isf(.05 / (2*local_len), local_len-2)) ** 2
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
        super(Derivative, self).__init__(input_series, window_size)
        self.sigma = sigma if sigma else DERIV_SIGMA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return np.nan
        local_series_diff = local_series.diff()
        local_series_diff[0] = 0.
        curr_pt = self.series.iloc[loc]
        curr_pt_diff = curr_pt - local_series.iloc[-1]

        z = abs((curr_pt_diff - local_series_diff.mean()) / curr_pt_diff.std())
        if z > self.sigma:
            return 1
        return 0

    def series_detect(self):
        self._series_detect()


class MedAbsDev(Prototype):
    def __init__(self, input_series, window_size=None, sigma=None):
        self.label = "MAD"
        window_size = window_size if window_size else MAD_WINDOW_SIZE
        super(MedAbsDev, self).__init__(input_series, window_size)
        self.sigma = sigma if sigma else MAD_SIGMA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return np.nan
        median = local_series.median()
        curr_pt = self.series.iloc[loc]
        curr_pt_demedianed = np.abs(curr_pt - median)
        med_abs_dev = sm.robust.mad(local_series)
        if med_abs_dev == 0:
            return np.nan
        test_stat = curr_pt_demedianed / med_abs_dev
        if test_stat > self.sigma:
            return 1
        return 0

    def series_detect(self):
        self._series_detect()


class kNNCAD(Prototype):
    def __init__(self, input_series, window_size=None, k=None, train_size=None, calibration_size=None, alpha=None):
        self.label = "KNN"
        window_size = window_size if window_size else KNN_WINDOW_SIZE
        self.k = k if k else KNN_K
        super(kNNCAD, self).__init__(input_series, window_size)
        self.train_size = train_size if train_size else int(self.window_size / 3)
        self.calibration_size = calibration_size if calibration_size else int(self.window_size / 3)
        self.covar_matrix = None
        self.alpha = alpha if alpha else KNN_ALPHA

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size+1, loc+1)
        if local_series is None: return np.nan
        transformed_curr_pt = local_series.iloc[-self.train_size:]
        train_matrix = []
        calibration_seq = []
        for i in range(0, 2 * self.train_size):
            if i < self.train_size: train_matrix.append(local_series.iloc[i:i+self.train_size])
            else: calibration_seq.append(local_series.iloc[i:i+self.train_size])
        train_matrix = np.array(train_matrix)
        try: self.covar_matrix = np.linalg.inv(np.dot(train_matrix.T, train_matrix))
        except np.linalg.LinAlgError:
            logging.error("Non-invertible covariance matrix - kNN aborted")
            return np.nan
        scores = np.array(map(lambda v: self._get_k_nearest_score(v, train_matrix), calibration_seq))
        curr_score = self._get_k_nearest_score(transformed_curr_pt, train_matrix)
        res = np.where(scores < curr_score)[0] / len(scores)
        return 1 if res > 1 - self.alpha else 0

    def series_detect(self):
        self.all_status = pd.Series(0, index=self.series.index)
        train_subseq = []
        calibration_subseq = []
        scores = []
        for i in range(0, len(self.series)):
            if i <= self.train_size:
                continue
            elif i <= 2 * self.train_size:
                transformed_ith_pt = self.series.iloc[i-self.train_size:i]
                train_subseq.append(transformed_ith_pt)
            else:
                transformed_ith_pt = self.series.iloc[i-self.train_size:i]
                ost = i % (2 * self.train_size)
                if ost == 0 or ost == int(self.train_size):
                    try: self.covar_matrix = np.linalg.inv(np.dot(np.array(train_subseq).T, np.array(train_subseq)))
                    except np.linalg.LinAlgError:
                        logging.error("Non-invertible covariance matrix at location" + str(i))
                        continue
                if len(scores) == 0:
                    scores = map(lambda v: self._get_k_nearest_score(v, np.array(train_subseq)), train_subseq)
                ith_score = self._get_k_nearest_score(transformed_ith_pt, train_subseq)
                ith_res = np.where(scores < ith_score)[0] / len(scores)
                self.all_status.iloc[i] = 1 if ith_res > 1 - self.alpha else 0

                if i >= 4 * self.train_size:
                    train_subseq.pop(0)
                    train_subseq.append(calibration_subseq.pop(0))

                scores.pop(0)
                calibration_subseq.append(transformed_ith_pt)
                scores.append(ith_score)
        self.processed = True
        logging.info("Pulse.kNNCAD: Series Processed")

    def _get_mahanlanobis_dist(self, a, b):
        assert len(a) == len(b)
        diff = a - np.array(b)
        return np.dot(np.dot(diff, self.covar_matrix), diff.T)

    def _get_k_nearest_score(self, item, reference):
        arr = map(lambda x: self._get_mahanlanobis_dist(x, item), reference)
        return np.sum(np.partition(arr, self.k)[:self.k])
