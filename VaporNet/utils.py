import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import namedtuple
from itertools import product
# dataset
from torch.utils.data import Dataset
# dataloader
from torch.utils.data import DataLoader
import pandas as pd


def sliding_windows(data, seq_length, frame_rate, video_class, Tsat, mf):
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
    Tsat:
        steady state saturation temperature
    mf:
        steady state mass flow rate

    Returns
    -------
    x
        formatted data
    y
        target output classes for each sequence
    T
        formatted Tsat for each sequence
    m
        formatted mass flow rate for each sequence
    """
    x = []  # truncated sequence sections
    y = []  # class label
    T = []
    m = []

    if(data.shape[0] >= seq_length * frame_rate):
        for i in range(len(data) - frame_rate * seq_length + 1):
            _x = data[i:(i + seq_length * frame_rate):frame_rate]
            x.append(_x)
            y.append(video_class)
            T.append(Tsat * np.ones((1, seq_length)))
            m.append(mf * np.ones((1, seq_length)))
    else:
        for i in range(len(data) - seq_length):
            _x = data[i:i + seq_length]
            x.append(_x)
            y.append(video_class)
            T.append(Tsat * np.ones((1, seq_length)))
            m.append(mf * np.ones((1, seq_length)))

    return np.array(x), np.array(y), np.array(T), np.array(m)


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

    def _get_label(self, idx):
        return self.y[idx]


def data_formatter(sequence_length, frame_rate, test_train='train'):
    '''Batch data formatting

    Parameters
    ----------
    seq_length : int
        desired sequence length for the LSTM
    frame_rate: int
        used to artificially simulate different frame rates by
        skipping frames within sequences
    test_train: str
        determines if function being used for testing or training

    Returns
    -------
    dataset
        formatted pytorch dataset
    input_size
        number of datapoints in set
    '''
    x = np.array([])
    y = np.array([])

    training_file = open(
        f'VaporNet/data/feature_sets/vapornet_{test_train}.csv', 'r')
    df = pd.read_csv(
        f'VaporNet/data/feature_sets/vapornet_{test_train}.csv',
        header=False)
    feature_set_vq = np.array([])
    x = torch.zeros(
        (round(
            df.shape[0] /
            sequence_length),
            sequence_length,
            df.shape[1] -
            2)).float()
    del df
    x_ind = 0
    y = torch.tensor([])
    vq = 2  # initialize vq

    # Each line contains a frames vapor quality, features, mass flow rate, and
    # saturation temperature
    for l, line in enumerate(training_file):

        # Keeps track of all frame features from the current video, for a given
        # vq
        feature_set_vq = np.concatenate((feature_set_vq, np.array([[float(i) for i in line.split(
            ',')[0:-4]]]))) if feature_set_vq.size else np.array([[float(i) for i in line.split(',')[0:-4]]])

        # Current lines vapor quality. Used to determine when data has moved
        # onto a new video
        vq = float(line.split(',')[-1])

        # Reset condition for new batch
        if(prev_vq == 2):
            prev_vq = vq
            Tsat = float(line.split(',')[-3])
            mf = float(line.split(',')[-2])

        # When new vq found, update x and y (data and targets) with everything
        # from last video
        elif(prev_vq != vq):
            if ((len(feature_set_vq) > sequence_length * frame_rate) & (l != 0)):
                feature_set_vq, training_labels_vq, training_Tsat_vq, training_mf_vq = sliding_windows(
                    feature_set_vq, sequence_length, frame_rate, vq, Tsat, mf)
                # New batch to be added to data
                batch = torch.cat((torch.tensor(feature_set_vq).squeeze(),
                                   torch.tensor(
                                       training_Tsat_vq).squeeze().unsqueeze(2),
                                   torch.tensor(training_mf_vq).squeeze().unsqueeze(2)), dim=2)

                if (x[x_ind:x_ind + batch.shape[0], :, :].shape == batch.shape):
                    x[x_ind:x_ind + batch.shape[0], :, :] = batch
                    # Keep track of where we are in the csv being read
                    x_ind = x_ind + batch.shape[0]
                    y = torch.cat(
                        (y.float(), torch.tensor(training_labels_vq).float()))
            # Start new batch
            vq = 2
            feature_set_vq = np.array([])
            continue

    training_file.close()
    input_size = x.shape[2] - 2
    x = x[:y.shape[0]]
    data_set = timeseries(x, y)
    return data_set, input_size
