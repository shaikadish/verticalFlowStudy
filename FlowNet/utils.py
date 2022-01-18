import numpy as np
import torch
from collections import namedtuple
from itertools import product
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def sliding_windows(data, seq_length, frame_rate, video_class):
    """Formats a set of data into the desired sequence of features

    Parameters
    ----------
    data : numpy array
        input data to be formatted
    seq_length : int
        desired sequence length for the LSTM
    frame_rate: int
        used to artificially simulate different frame rates by
        skipping frames within sequences
    video_class: int
        target output class for sequence

    Returns
    -------
    x
        formatted data
    y
        target output classes for each sequence
    """

    x = []  # truncated sequence sections
    y = []  # class label

    if(data.shape[0] > seq_length * frame_rate):
        for i in range(len(data) - frame_rate * seq_length - 1):
            _x = data[i:(i + seq_length * frame_rate):frame_rate]
            x.append(_x)
            y.append(video_class)
    else:
        for i in range(len(data) - seq_length - 1):
            _x = data[i:i + seq_length]
            x.append(_x)
            y.append(video_class)

    return np.array(x), np.array(y)


def data_formatter(sequence_length, frame_rate, test_train='train'):
    '''Batch execution of sliding windows

    Returns
    -------
    dataset
        formatted pytorch dataset
    input_size
        number of datapoints in set
    '''

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = np.array([])
    y = np.array([])
    path = 'data/feature_sets/'
    for i in range(0, 8):

        unformatted_data = np.loadtxt(
            path + f's{i}_{test_train}.csv', delimiter=',')
        data, labels = sliding_windows(
            unformatted_data.squeeze(), sequence_length, frame_rate, i)

        x = np.concatenate((x, data)) if x.size else data
        y = np.concatenate((y, labels))

    input_size = x.shape[2]
    data_set = timeseries(scaler.fit_transform(torch.tensor(
        x).reshape(-1, 1)).reshape(-1, sequence_length, input_size), y)
    return data_set, input_size


class RunBuilder():
    '''Class used to generate runs during hyper parameter search'''
    @staticmethod
    def get_runs(params):
        # creates Run class to encapsulate data
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len
