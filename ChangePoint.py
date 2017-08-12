from Prototype import *
import numpy as np
from scipy import stats
from settings import *


class TTest(Prototype):
    """
    Two sample Welch's T-test: determine step change in mean
    """
    def __init__(self, input_series, window_size=None, probe_size=None, alpha=None):
        self.label = "TTest"
        window_size = window_size if window_size else 15
        probe_size = probe_size if probe_size else 5
        if probe_size >= window_size:
            raise Exception("Probe size must be smaller than window size")
        super(TTest, self).__init__(input_series, window_size)
        self.probe_size = probe_size
        self.alpha = alpha if alpha else 0.05

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size+self.probe_size, loc+1)
        if local_series is None: return np.nan
        reference = local_series.iloc[:self.window_size]
        probe = local_series.iloc[self.window_size:]
        t_stat, p_val = stats.ttest_ind(reference, probe, equal_var=False)
        if abs(t_stat) > 4 and p_val < self.alpha:
            return 1 if t_stat < 0 else -1
        return 0

    def series_detect(self):
        self._series_detect()


class FTest(Prototype):
    """
    Two-sample F-test: determine step change in variance
    """
    def __init__(self, input_series, window_size=None, probe_size=None, alpha=None):
        self.label = "FTest"
        window_size = window_size if window_size else 15
        probe_size = probe_size if probe_size else 5
        if probe_size > window_size:
            raise Exception("Probe size must be smaller than window size")
        super(FTest, self).__init__(input_series, window_size)
        self.probe_wize = probe_size
        self.alpha = alpha if alpha else 0.05

    def point_detect(self, loc):
        local_series = self._get_local_series(self.series, self.window_size+self.probe_wize, loc+1)
        if local_series is None: return np.nan
        diff_series = self._get_diff_series(local_series)
        _, normal_p_val = stats.normaltest(diff_series)
        if normal_p_val > 0.05:
            # F-test is sensitive to normal distribution assumption. Check the normalcy of the differenced series.
            return np.nan
        reference = local_series.iloc[:self.window_size]
        probe = local_series.iloc[self.window_size:]

        probe_var = probe.var()
        ref_var = reference.var()
        F = probe_var / ref_var
        df_1 = self.probe_wize - 1
        df_2 = self.window_size - 1
        p_val = 1 - stats.f.cdf(F, df_1, df_2)
        return 1 if p_val < self.alpha else 0

    def series_detect(self):
        self._series_detect()

    @staticmethod
    def _get_diff_series(series):
        return series.diff().dropna()