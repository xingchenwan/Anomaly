from settings import *
import pandas as pd
import logging
import Pulse as Pulse
import ChangePoint as Trend
import Assembler

def analyze_entire_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        logging.critical("Invalid Input Frame")
        return

    pulse_res, trend_res = [], []
    for col in frame.columns:
        series = frame[col]
        pulse, trend = analyze_series(series)
        pulse_res.append(pulse)
        trend_res.append(trend)


def analyze_series(series_input):
    if not isinstance(series_input, pd.Series):
        logging.critical("Invalid Input Series")
        return
    res_list = []
    if RUN_KS: res_list.append(Pulse.KS(series_input).get_all_status())
    if RUN_DERIV: res_list.append(Pulse.Derivative(series_input).get_all_status())
    if RUN_GRUBBS: res_list.append(Pulse.Grubbs(series_input).get_all_status())
    if RUN_PEWMA: res_list.append(Pulse.ProbabilisticEWMA(series_input).get_all_status())
    if RUN_SMA: res_list.append(Pulse.SimpleMovingAvg(series_input).get_all_status())
    if RUN_MAD: res_list.append(Pulse.MedAbsDev(series_input).get_all_status())

    pulse_base = pd.concat(res_list, axis=1) if len(res_list) else None
    pulse_assembled = Assembler.pulse_assembler(pulse_base, mode='vote_abs', threshold=3)

    if RUN_ZSCORE: res_list.append(Trend.ZScore(series_input).get_all_status())
    trend_assembled = pd.DataFrame(res_list, axis=1) if len(res_list) else None

    return pulse_assembled, trend_assembled