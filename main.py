import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Detectors.Pulse as Pulse
import logging
import Analyzer

def load_csv():
    file_path = "input/datatraining.txt"
    frame = pd.read_csv(file_path)
    frame = frame[["Temperature", "Humidity", "Light", "CO2"]]
    return frame

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frame = load_csv()
    test_series = frame["Humidity"].iloc[0:100]


    res = Analyzer.analyze_series(test_series, 4)[0]
    for index, row in res.iteritems():
        print(index, row)
