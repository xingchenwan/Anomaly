import pandas as pd
import logging


class Prototype(object):
    def __init__(self, input_series, window_size):
        if not isinstance(input_series, pd.Series):
            raise Exception("Unknown Input Format")
        self.series = input_series
        self.window_size = window_size
        self.processed = False

    def point_detect(self, loc):
        pass

    def series_detect(self):
        pass

    def _series_detect(self):
        self.all_status = pd.Series(0, index=self.series.index)
        ct = 0
        for i in range(0, len(self.series)):
            res = self.point_detect(i)
            self.all_status.iloc[i] = res
            if res != 0: ct += 1
        self.processed = True
        logging.info("Proportion of anomalous points:" + str(ct / len(self.series)))

    def get_all_status(self):
        if not self.processed: self.series_detect()
        self.all_status.name = self.series.name + "_" + self.label
        return self.all_status

    @staticmethod
    def _get_local_series(series, length, loc):
        if loc < 0: loc = len(series) + loc
        if loc - length <= 0: return None
        local_series = series.iloc[loc - length:loc]
        half_len = int(length/2)
        if local_series[-half_len:].value_counts().iloc[0] / len(local_series) > 1/3:
            logging.warning("Boring Series Detected.")
            return None
        return local_series
