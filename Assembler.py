import pandas as pd
import numpy as np
import logging


def pulse_assembler(pulse_base, mode, threshold=3):
    if not isinstance(pulse_base, pd.DataFrame):
        logging.critical("Unknown assembler input")
        return None

    if mode == 'vote_abs':
        if not isinstance(threshold, int):
            logging.error("Enter an integer value of detection threshold for mode vote_abs")
            return None
        res = pd.Series(0, index=pulse_base.index)
        for index, item in pulse_base.iterrows():
            if (abs(item) == 1).any():
                if len([x for x in item[abs(item) == 1]]) >= threshold:
                    res[index] = 1

    elif mode == 'vote_pct':
        if not (threshold > 0. and threshold < 1.):
            logging.error("Enter a float value of detection threshold for mode vote_pct")
            return None
        res = pd.Series(0, index=pulse_base.index)
        for index, item in pulse_base.iterrows():
            if (abs(item) == 1).any():
                non_nan_vals = len([x for x in item[item != np.nan]])
                thres = int(non_nan_vals * threshold)
                if len([x for x in item[abs(item) == 1]]) >= thres:
                    res[index] = 1

    elif mode == 'logistic':
        pass

    else:
        logging.error("Unknown assembler argument")
        return None