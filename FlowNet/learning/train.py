"""FlowNet training

This script trains FlowNet using 5-fold training, and saves the various models.
This script also facilitates an appropriate hyper-parameter search.
"""

# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from model import LSTM
from sklearn.model_selection import KFold

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kfold testing configuration
kfold = KFold(n_splits=5, shuffle=True)

# Dictionary of Hyper-parameters tested during training
params = dict(
    learning_rate=[1e-4], batch_size=[256], num_epochs=[50], sequence_length=[20, 50], hidden_size=[256], num_layers=[2], frame_rate=[25, 30]
)

# Each combination of Hyper-parameters is tested during training, and
# counts as a run
run_counter = 0
for run in RunBuilder.get_runs(params):

    # Run configuration
    num_classes = 8
    num_epochs = run.num_epochs
    batch_size = run.batch_size
    learning_rate = run.learning_rate
    sequence_length = run.sequence_length
    hidden_size = run.hidden_size
    num_layers = run.num_layers
    frame_rate = run.frame_rate

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
    print("Frame Rate:          {}".format(frame_rate))

    for fold, (train_data, val_data) in enumerate(kfold.split(dataset)):
        print("-------Fold {}/5------".format(fold + 1))

        # Load in training and validation data for fold
        train_set = torch.utils.data.SubsetRandomSampler(train_data)
        val_set = torch.utils.data.SubsetRandomSampler(val_data)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        # Max validation accuracy stored to save only most accurate models for
        # each fold
        max_accuracy = 0

        # Create model based on run parameters
        model = LSTM(
            input_size,
            hidden_size,
            num_layers,
            num_classes).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            training_preds = torch.tensor([])
            training_labels = torch.tensor([])
            for i, (data, labels) in enumerate(train_loader):

                # Data and labels for batch
                data = data.to(device)
                labels = labels.long().to(device)
                training_labels = torch.cat(
                    (training_labels, torch._cast_Float(labels.cpu())))

                # Forward pass
                outputs = model(data)
                training_preds = torch.cat(
                    (training_preds, torch._cast_Float(
                        outputs.argmax(
                            dim=1).cpu())))
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
                correct = 0
                for j, (val_data, val_labels) in enumerate(val_loader):
                    model.eval()

                    # Data and labels for batch
                    val_data = val_data.to(device)
                    val_labels = val_labels.long().to(device)

                    # Forward pass
                    val_outputs = model(val_data)
                    val_loss = criterion(val_outputs, val_labels)

                    val_preds = torch.max(val_outputs, 1)[1]
                    correct += sum(val_labels.cpu().numpy()
                                   == val_preds.cpu().numpy())
                    val_accuracy = 100 * correct / ((j + 1) * batch_size)
                print('Validation Accuracy: {}'.format(
                    100 * correct / ((j + 1) * batch_size)))

                # Save the highest performing model itteration for a given fold
                if(val_accuracy > max_accuracy):
                    torch.save(
                        model.state_dict(),
                        'Flownet/data/pretrained_models/all_models/LSTM_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(
                            sequence_length,
                            hidden_size,
                            num_layers,
                            epoch,
                            round(val_accuracy),
                            fold,
                            loss.item(),
                            frame_rate))
                    max_accuracy = 100 * correct / ((j + 1) * batch_size)

                model.train()
