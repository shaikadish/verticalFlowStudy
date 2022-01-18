"""FrameNet testing

This script runs the 5 FrameNet models generated during 5-fold training
on the test batch, and formats and saves their outputs for further evaluation.
"""

# Imports
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder
from torchvision.models.resnet import *
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data processing definition
transform = transforms.Compose([
    transforms.CenterCrop((512, 140)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load in hyper parameters for 5-fold test
# Run being tested must have all 5 models in the 'test_models' folder
path = 'FrameNet/data/pretrained_models/test_models'
network_paths = sorted(os.listdir(os.path.join(os.getcwd(), path)))

num_classes = 7
batch_size = 256

# Format dataset for run. Dataset will be different depending on sequence
# length and frame rate
dataset = DatasetFolder(
    root=f'FrameNet/data/frame_class_images/test',
    transform=transform)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load in model parameters
model0 = resnet101(num_classes=num_classes).to(device)
model1 = resnet101(num_classes=num_classes).to(device)
model2 = resnet101(num_classes=num_classes).to(device)
model3 = resnet101(num_classes=num_classes).to(device)
model4 = resnet101(num_classes=num_classes).to(device)

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
criterion = nn.CrossEntropyLoss()

# Test all 5 models at once, saving the outputs
lines = torch.tensor([])
for i, (data, labels) in enumerate(test_loader):

    # Load in batch being tested
    data = data.to(device)
    labels = labels.long().to(device)

    # Get desired outputs from each fold
    outputs0 = model0(data).squeeze()
    loss0 = criterion(outputs0, labels)
    outputs0 = torch.max(outputs0, 1)[1]

    outputs1 = model1(data).squeeze()
    loss1 = criterion(outputs1, labels)
    outputs1 = torch.max(outputs1, 1)[1]

    outputs2 = model2(data).squeeze()
    loss2 = criterion(outputs2, labels)
    outputs2 = torch.max(outputs2, 1)[1]

    outputs3 = model3(data).squeeze()
    loss3 = criterion(outputs3, labels)
    outputs3 = torch.max(outputs3, 1)[1]

    outputs4 = model4(data).squeeze()
    loss4 = criterion(outputs4, labels)
    outputs4 = torch.max(outputs4, 1)[1]

    # Save all outputs
    new_line = torch.stack(
        (labels,
         outputs0,
         outputs1,
         outputs2,
         outputs3,
         outputs4),
        1)
    lines = torch.cat((lines.long(), new_line))
    new_line_loss = torch.tensor([loss0.item() /
                                  batch_size, loss1.item() /
                                  batch_size, loss2.item() /
                                  batch_size, loss3.item() /
                                  batch_size, loss4.item() /
                                  batch_size])
    lines_loss = torch.cat((lines_loss, new_line_loss.unsqueeze(0)))

np.savetxt("FrameNet/data/outputs/FrameNet_kfold.csv",
           lines.detach().numpy(), delimiter=',')
np.savetxt("FrameNet/data/outputs/FrameNet_loss_kfold.csv",
           lines_loss.detach().numpy(), delimiter=',')
