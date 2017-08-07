from settings import *
import pandas as pd
import logging
import Detectors.UnivariatePulse as Pulse
import Detectors.UnivariateTrend as Trend


def analyze_series(series_input, threshold):
    if isinstance(series_input, pd.Series):
        logging.critical("Invalid Input Series")
        return
    res_list = []
    if RUN_KS: res_list.append(Pulse.KS(series_input).get_all_status())
    if RUN_DERIV: res_list.append(Pulse.Derivative(series_input).get_all_status())
    if RUN_GRUBBS: res_list.append(Pulse.Grubbs(series_input).get_all_status())
    if RUN_PEWMA: res_list.append(Pulse.ProbabilisticEWMA(series_input).get_all_status())
    if RUN_SMA: res_list.append(Pulse.SimpleMovingAvg(series_input).get_all_status())
    pulse_base = pd.concat(res_list, axis=1)
    pulse_assembled = pd.Series(0, index=series_input.index)
    
    for index, item in pulse_base.iterrows():
        if (item != 0).any():
            if len((item != 0).any()) >= threshold:
                pulse_assembled[index] = 1
                
    res_list = []
    if RUN_ZSCORE: res_list.append(Trend.ZScore(series_input).get_all_status())
    trend_assembled = pd.concat(res_list, axis=1)
    
    return pulse_assembled, trend_assembled