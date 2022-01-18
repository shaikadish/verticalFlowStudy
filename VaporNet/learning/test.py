"""VaporNet testing

This script runs the 5 VaporNet models generated during 5-fold training
on the test batch, and formats and saves their outputs for further evaluation.
"""

# Imports
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from utils import *
from model import LSTM_regression as LSTM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load in hyper parameters for 5-fold test
# Run being tested must have all 5 models in the 'test_models' folder
path = 'VaporNet/data/pretrained_models/test_models'
network_paths = sorted(os.listdir(os.path.join(os.getcwd(), path)))
# Run parameters are in the model names
run_params = network_paths[0].split('_')
num_classes = 1
sequence_length = run_params[2]
hidden_size = run_params[3]
num_layers = run_params[4]
num_flat = run_params[5]
frame_rate = run_params[-1]
batch_size = 256

# Format dataset for run. Dataset will be different depending on sequence
# length and frame rate
dataset, input_size = data_formatter(
    sequence_length, frame_rate, test_train='test')
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load in model parameters
model0 = LSTM(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    num_flat).to(device)
model1 = LSTM(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    num_flat).to(device)
model2 = LSTM(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    num_flat).to(device)
model3 = LSTM(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    num_flat).to(device)
model4 = LSTM(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    num_flat).to(device)

model0.load_state_dict(torch.load(
    path + '/{}'.format(network_paths[0]), map_location=torch.device(device)))
model1.load_state_dict(torch.load(
    path + '/{}'.format(network_paths[1]), map_location=torch.device(device)))
model2.load_state_dict(torch.load(
    path + '/{}'.format(network_paths[2]), map_location=torch.device(device)))
model3.load_state_dict(torch.load(
    path + '/{}'.format(network_paths[3]), map_location=torch.device(device)))
model4.load_state_dict(torch.load(
    path + '/{}'.format(network_paths[4]), map_location=torch.device(device)))

model0.eval()
model1.eval()
model2.eval()
model3.eval()
model4.eval()

# Loss function
criterion = nn.MSELoss()

# Test all 5 models at once, saving the outputs
lines = torch.tensor([])
lines_loss = torch.tensor([])
for i, (data, labels) in enumerate(test_loader):

    # Load in batch being tested
    data = data.to(device)
    labels = labels.to(device)

    # Get desired outputs from each fold
    outputs0 = model0(data[:, :, 0:-2], data[:, 0, -2],
                      data[:, 0, -1]).squeeze()
    outputs1 = model1(data[:, :, 0:-2], data[:, 0, -2],
                      data[:, 0, -1]).squeeze()
    outputs2 = model2(data[:, :, 0:-2], data[:, 0, -2],
                      data[:, 0, -1]).squeeze()
    outputs3 = model3(data[:, :, 0:-2], data[:, 0, -2],
                      data[:, 0, -1]).squeeze()
    outputs4 = model4(data[:, :, 0:-2], data[:, 0, -2],
                      data[:, 0, -1]).squeeze()

    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    loss3 = criterion(outputs3, labels)
    loss4 = criterion(outputs4, labels)

    # Save all outputs
    new_line = torch.stack(
        (labels,
         outputs0,
         outputs1,
         outputs2,
         outputs3,
         outputs4),
        1)
    lines = torch.cat((lines, new_line))
    new_line_loss = torch.tensor([loss0.item() /
                                  batch_size, loss1.item() /
                                  batch_size, loss2.item() /
                                  batch_size, loss3.item() /
                                  batch_size, loss4.item() /
                                  batch_size])
    lines_loss = torch.cat((lines_loss, new_line_loss.unsqueeze(0)))

np.savetxt("VaporNet/data/outputs/VaporNet_kfold.csv",
           lines.detach().numpy(), delimiter=',')
np.savetxt("VaporNet/data/outputs/VaporNet_loss_kfold.csv",
           lines_loss.detach().numpy(), delimiter=',')
