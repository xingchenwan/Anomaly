from Detectors.Prototype import *
import numpy as np
from settings import *


class ZScore(Prototype):
    def __init__(self, input_series, window_size=None, probe_size=None, sigma=None):
        self.label = "ZSCORE"
        window_size = window_size if window_size else ZSCORE_WINDOW_SIZE
        probe_size = probe_size if probe_size else ZSCORE_PROBE_SIZE
        if probe_size >= window_size:
            raise Exception("Probe size must be smaller than window size")
        super(ZScore).__init__(input_series, window_size)
        self.probe_size = probe_size
        self.sigma = sigma if sigma else ZSCORE_SIGMA

    def point_detect(self, loc):
        reference_window = self._get_local_series(self.series, self.window_size, loc-self.probe_size)
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




