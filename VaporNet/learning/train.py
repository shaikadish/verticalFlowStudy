"""VaporNet training

This script trains VaporNet using 5-fold training, and saves the various models.
This script also facilitates an appropriate hyper-parameter search.
"""

# Imports
import torch
import torch.nn as nn
from model import LSTM_regression as LSTM
from torch.utils.data import DataLoader
from utils import *
from sklearn.model_selection import KFold

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kfold testing configuration
kfold = KFold(n_splits=5, shuffle=True)

# Dictionary of Hyper-parameters tested during training
params = dict(
    learning_rate=[1e-3], batch_size=[256], num_epochs=[150], sequence_length=[50], hidden_size=[256], num_layers=[1], frame_rate=[10, 15], num_flat=[2]
)

# Each combination of Hyper-parameters is tested during training, and
# counts as a run
run_counter = 0
for run in RunBuilder.get_runs(params):

    # Run Configuration
    num_classes = 1
    num_epochs = run.num_epochs
    batch_size = run.batch_size
    learning_rate = run.learning_rate
    sequence_length = run.sequence_length
    hidden_size = run.hidden_size
    num_layers = run.num_layers
    frame_rate = run.frame_rate
    num_flat = run.num_flat

    # Format dataset for run. Dataset will be different depending on sequence
    # length and frame rate
    dataset, input_size = data_formatter(sequence_length, frame_rate)

    print("-------Run number {}/{}------".format(run_counter,
          len(RunBuilder.get_runs(params)) - 1))
    run_counter += 1
    print("Number of Epochs:    {}".format(num_epochs))
    print("Batch Size:          {}".format(batch_size))
    print("Learning Rate:       {}".format(learning_rate))
    print("Sequence Length:     {}".format(sequence_length))
    print("Hidden Layer size:   {}".format(hidden_size))
    print("Number Layers:       {}".format(num_layers))
    print("Number Flat Layers:  {}".format(num_flat))
    print("Frame Rate:          {}".format(frame_rate))

    for fold, (train_data, val_data) in enumerate(kfold.split(dataset)):
        print("-------Fold {}/5------".format(fold + 1))

        # Load in training and validation data for fold
        train_set = torch.utils.data.SubsetRandomSampler(train_data)
        val_set = torch.utils.data.SubsetRandomSampler(val_data)

        # ,sampler=ImbalancedDatasetSampler(training_set)) #SAMPLER REPLACED SHUFFLE
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)

        # Min validation loss is stored to save only the best model for each
        # fold
        val_loss_min = 100

        # Create model based on run parameters
        model = LSTM(
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            num_flat).to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            training_preds = torch.tensor([])
            training_labels = torch.tensor([])
            for i, (data, labels) in enumerate(train_loader):

                # Data and labels for batch
                data = data.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(data[:, :, 0:-2], data[:, 0, -2],
                                data[:, 0, -1]).squeeze()  # multi channel
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 50 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

            # Validation
            if (epoch + 1) % 2 == 0:
                model.eval()
                for j, (val_data, val_labels) in enumerate(val_loader):

                    # Data and labels for batch
                    val_data = val_data.to(device)
                    val_labels = val_labels.to(device)

                    # Forward pass
                    val_outputs = model(
                        val_data[:, :, 0:-2], val_data[:, 0, -2], val_data[:, 0, -1]).squeeze()
                    val_loss = criterion(val_outputs, val_labels)

                    val_preds = val_outputs

                # Run data for high level analysis
                print('Validation Loss: {}'.format(val_loss.item()))
                print(
                    "Validation Predictions: {}".format(
                        val_preds.squeeze()[
                            0:30]))
                print("Validation labels: {}".format(val_labels[0:30]))

                # Save the highest performing model itteration for a given fold
                if(val_loss.item() < val_loss_min):
                    val_loss_min = val_loss.item()
                    save_sl = sequence_length
                    save_hs = hidden_size
                    save_nl = num_layers
                    save_fl = num_flat
                    save_model = model.state_dict()

                model.train()

        torch.save(
            save_model,
            'VaporNet/data/pretrained_models/all_models/LSTM_r_{}_{}_{}_{}_{}_{}_{}.pth'.format(
                save_sl,
                save_hs,
                save_nl,
                save_fl,
                val_loss_min,
                fold,
                frame_rate))
