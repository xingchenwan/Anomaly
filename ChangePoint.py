from Prototype import *
import numpy as np
from scipy import stats
from settings import *
from scipy.special import gammaln


class ZScore(Prototype):
    def __init__(self, input_series, window_size=None, probe_size=None, sigma=None):
        self.label = "ZSCORE"
        window_size = window_size if window_size else ZSCORE_WINDOW_SIZE
        probe_size = probe_size if probe_size else ZSCORE_PROBE_SIZE
        if probe_size >= window_size:
            raise Exception("Probe size must be smaller than window size")
        super(ZScore, self).__init__(input_series, window_size)
        self.probe_size = probe_size
        self.sigma = sigma if sigma else ZSCORE_SIGMA

    def point_detect(self, loc, mode='static'):
        if mode == 'static':
            reference_window = self._get_local_series(self.series, self.window_size, loc-self.probe_size)
        elif mode == 'adaptive':
            ref_win_size = self._get_ref_win_size(loc)
            if ref_win_size < self.window_size:  # Insufficient information
                return 0
            reference_window = self._get_local_series(self.series, ref_win_size, loc-self.probe_size)
        else:
            raise Exception("Unknown point detect mode.")
        probe_window = self._get_local_series(self.series, self.probe_size, loc)
        if reference_window is None or probe_window is None:
            return 0
        reference_mean = reference_window.mean()
        probe_len = len(probe_window)
        probe_mean = probe_window.mean()
        sterr = reference_window.std() / np.sqrt(probe_len)
        zscore = abs((probe_mean - reference_mean) / sterr)
        if zscore > self.sigma:
            return 1
        return 0

    def series_detect(self):
        self._series_detect()

    def _get_ref_win_size(self, loc):
        # Distance between current point and last identified change point
        ref_win_size = 0
        for pt in range(loc-1, -1, -1):
            ref_win_size += 1
            if self.point_detect(pt,mode='static') != 0:
                break
        return ref_win_size


class Bayesian(Prototype):
    def __init__(self, input_series, window_size=None, probe_size=None,
                 theta_alpha=None, theta_beta=None, theta_kappa=None,
                 theta_mu=None):
        self.label = "BAYESIAN"
        window_size = window_size if window_size else BAYESIAN_WINDOW_SIZE
        probe_size = probe_size if probe_size else BAYESIAN_PROBE_SIZE
        if probe_size >= window_size:
            raise Exception("Probe size must be smaller than window size")
        super(Bayesian, self).__init__(input_series, window_size)
        self.probe_size = probe_size

    def point_detect(self, loc):
        curr_pt = self.series.iloc[loc]
        local_series = self._get_local_series(self.series, self.window_size, loc)
        if local_series is None: return 0
        n = len(local_series)

        for t in range(n):
            g[t] = np.log(prior_func(t))
            if t == 0:
                G[t] = g[t]
            else:
                G[t] = np.logaddexp(G[t - 1], g[t])

        P[n - 1, n - 1] = observation_log_likelihood_function(data, n - 1, n)
        Q[n - 1] = P[n - 1, n - 1]

        pass

    def series_detect(self):
        self._series_detect()

    @staticmethod
    def _log_likelihood(series):
        n = len(series)
        mean = series.mean()

        mu = (n * mean) / (1 + n)
        nu = 1 + n
        alpha = 1 + n/2
        beta = 1 + 0.5*((series - mean)**2).sum(0) + (n/(1 + n)) * (mean**2 / 2)
        scale = (beta*(nu + 1)) / (alpha * nu)
        prob = np.sum(np.log(1 + (series - mu)**2/(nu*scale)))
        lgA = gammaln((nu + 1) / 2) - np.log(np.sqrt(np.pi * nu * scale)) - gammaln(nu / 2)
        return np.sum(n * lgA - (nu + 1) / 2 * prob)

