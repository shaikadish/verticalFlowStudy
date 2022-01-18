"""Frames to features conversion

This script uses FrameNet to convert a sequence of images to a sequence of
features for each flow regime class.

It can be used to generate a training or testing set, and saves the feature sequence
for class n in sn.csv
"""

# Imports
import torch
import torchvision.transforms as transforms
from learning.resnet_feature_output import *
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse

# Args. By default, will extract training data unless test argument specified
parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()
if args.test:
    test_train = 'test'
else:
    test_train = 'train'

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load in model parameters for FrameNet
path = 'FrameNet/data/pretrained_models/all_models'
model_params = '/model_name.pth'
resnet = resnet101(num_classes=7).to(device)
resnet.eval()
if os.path.isfile(path + model_params):
    resnet.load_state_dict(torch.load(path + model_params))
    print('\n[*]parameters loaded')

# Image data processing definition
transform = transforms.Compose([
    transforms.CenterCrop((512, 140)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize image data loader
dataset = DatasetFolder(
    root=f'FlowNet/data/regime_images/{test_train}',
    transform=transform)
loader = DataLoader(dataset=dataset,
                    batch_size=10,
                    num_workers=1,
                    shuffle=False)


@torch.no_grad()
def sequence_extractor():
    # Feature sequence for each video class
    s0 = torch.tensor([]).to('cpu')
    s1 = torch.tensor([]).to('cpu')
    s2 = torch.tensor([]).to('cpu')
    s3 = torch.tensor([]).to('cpu')
    s4 = torch.tensor([]).to('cpu')
    s5 = torch.tensor([]).to('cpu')
    s6 = torch.tensor([]).to('cpu')
    s7 = torch.tensor([]).to('cpu')
    for img, vid_class in loader:
        img = img.to(device)
        # Outputs raw image features used as input to FlowNet
        outputs = resnet(img)
        features = outputs.to('cpu')
        torch.cat((features, torch.tensor([vid_class])))
        torch.cuda.empty_cache()
        for j in range(len(vid_class)):
            if(vid_class[j] == 0):
                if (len(s0) == 0):
                    s0 = features[j].unsqueeze(dim=0)
                else:
                    s0 = torch.cat(
                        (torch.tensor(s0),
                         features[j].unsqueeze(
                            dim=0)))
            elif (vid_class[j] == 1):
                if (len(s1) == 0):
                    s1 = features[j].unsqueeze(dim=0)
                else:
                    s1 = torch.cat(
                        (torch.tensor(s1),
                         features[j].unsqueeze(
                            dim=0)))
            elif (vid_class[j] == 2):
                if (len(s2) == 0):
                    s2 = features[j].unsqueeze(dim=0)
                else:
                    s2 = torch.cat(
                        (torch.tensor(s2),
                         features[j].unsqueeze(
                            dim=0)))
            elif (vid_class[j] == 3):
                if (len(s3) == 0):
                    s3 = features[j].unsqueeze(dim=0)
                else:
                    s3 = torch.cat(
                        (torch.tensor(s3),
                         features[j].unsqueeze(
                            dim=0)))
            elif (vid_class[j] == 4):
                if (len(s4) == 0):
                    s4 = features[j].unsqueeze(dim=0)
                else:
                    s4 = torch.cat(
                        (torch.tensor(s4),
                         features[j].unsqueeze(
                            dim=0)))
            elif (vid_class[j] == 5):
                if (len(s5) == 0):
                    s5 = features[j].unsqueeze(dim=0)
                else:
                    s5 = torch.cat(
                        (torch.tensor(s5),
                         features[j].unsqueeze(
                            dim=0)))
            elif (vid_class[j] == 6):
                if (len(s6) == 0):
                    s6 = features[j].unsqueeze(dim=0)
                else:
                    s6 = torch.cat(
                        (torch.tensor(s6),
                         features[j].unsqueeze(
                            dim=0)))
            elif (vid_class[j] == 7):
                if (len(s7) == 0):
                    s7 = features[j].unsqueeze(dim=0)
                else:
                    s7 = torch.cat(
                        (torch.tensor(s7),
                         features[j].unsqueeze(
                            dim=0)))

    # Ordered feature outputs for each regime class in CSV format
    save_path = "FlowNet/data/regime_feature_sets/"
    np.savetxt(
        save_path +
        f's0_{test_train}.csv',
        s0.cpu().detach().numpy(),
        delimiter=',')
    np.savetxt(
        save_path +
        f's1_{test_train}.csv',
        s1.cpu().detach().numpy(),
        delimiter=',')
    np.savetxt(
        save_path +
        f's2_{test_train}.csv',
        s2.cpu().detach().numpy(),
        delimiter=',')
    np.savetxt(
        save_path +
        f's3_{test_train}.csv',
        s3.cpu().detach().numpy(),
        delimiter=',')
    np.savetxt(
        save_path +
        f's4_{test_train}.csv',
        s4.cpu().detach().numpy(),
        delimiter=',')
    np.savetxt(
        save_path +
        f's5_{test_train}.csv',
        s5.cpu().detach().numpy(),
        delimiter=',')
    np.savetxt(
        save_path +
        f's6_{test_train}.csv',
        s6.cpu().detach().numpy(),
        delimiter=',')
    np.savetxt(
        save_path +
        f's7_{test_train}.csv',
        s7.cpu().detach().numpy(),
        delimiter=',')


def main():
    sequence_extractor()


if __name__ == '__main__':
    main()
