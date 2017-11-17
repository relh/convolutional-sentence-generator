import numpy as np
import pandas as pd

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def preprocess(path):
    data = pd.read_csv(path, sep='\t').get_values()[:, 1:]

    c = np.diff(data[:, 0]) / (np.abs(data[:-1, 0])) * 100
    h = np.diff(data[:, 1]) / (np.abs(data[:-1, 1])) * 100
    l = np.diff(data[:, 2]) / (np.abs(data[:-1, 2])) * 100
    o = np.diff(data[:, 3]) / (np.abs(data[:-1, 3])) * 100
    qv = data[:, 4]
    v = data[:, 5]
    wa = np.diff(data[:, 6]) / (np.abs(data[:-1, 6])) * 100

    # Max
    for data in [c, h, l, o, qv, v, wa]:
      std_dev = np.std(data)
      print("Std Dev: \t" + str(std_dev))
      print("Max: \t\t" + str(max(data)))
      print("Min: \t\t" + str(min(data)))
      data[data > 3*std_dev] = 3*std_dev
      data[data < 3*-std_dev] = 3*-std_dev
      print(max(data))
      print(min(data))

    print(data)
    return data

