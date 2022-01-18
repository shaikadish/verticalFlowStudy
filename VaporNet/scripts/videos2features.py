"""Videos to feature set conversion

This script converts input videos to the set of features describing each image in the video.
These features are saved, along with the video name, mass flow rate, saturation temperature,
and target vapor quality for the set of features.
"""

# Imports
import torch
import torchvision.transforms as transforms
from learning.resnet_feature_output import *
import os
import cv2
import argparse


class process_data():
    def __init__(self, test_train='train'):

        # Use GPU if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load in model parameters for FrameNet
        framenet_path = 'FrameNet/data/pretrained_models/all_models/'
        model_name = 'model_name.pth'
        self.resnet = resnet101(num_classes=7).to(self.device)
        self.resnet.eval()
        if os.path.isfile(framenet_path + model_name):
            self.resnet.load_state_dict(torch.load(framenet_path + model_name))
            print('\n[*]parameters loaded')

        # Image data processing definition
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.CenterCrop((512, 140)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Create file for feature set output
        path = 'VaporNet/data/feature_sets'
        self.output_features = open(path + f'/vapornet_{test_train}.csv', 'w')

        # Create list of all videos being processed
        self.video_path = f'VaporNet/data/flow_videos/{test_train}'
        video_list = sorted(
            os.listdir(
                os.path.join(
                    os.getcwd(),
                    self.video_path)))

        # Process each video into a sequence of features describing the videos
        # sequence of frames
        for video_name in video_list:
            self.video2features(video_name)

        # Close the output file
        self.output_features.close()

    def video2features(self, video_name):

        # Load in and loop through each video frame by frame
        capture = cv2.VideoCapture(self.video_path + '/' + video_name)
        while True:
            ret, image = capture.read()
            if not ret:
                break

            # for exception when m instead of mf in video name
            try:
                Tsat = int(video_name.split('\\')[-1].split('=')[1][0:-3])
            except ValueError:
                Tsat = int(video_name.split('\\')[-1].split('=')[1][0:-2])

            mf = int(video_name.split('\\')[-1].split('=')[2][0:-2])
            vq = float(video_name.split('\\')[-1].split('=')[3][0:-2])

            # Extract a set of features for the frame
            img = self.transform(image).to(self.device)
            feature_output = self.resnet(
                img.unsqueeze(
                    dim=0)).detach().squeeze().tolist()

            # For each frame, append output features, video name, saturation
            # temperature, mass flow rate, and target vapor quality to the
            # output file
            features_string = str(feature_output)[1:-1]
            self.output_features.write('{},{},{},{},{}\n'.format(
                features_string, video_name.split('\\')[-1], Tsat, mf, vq))


def main():
    # Args. By default, will extract training data unless test argument
    # specified
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    if args.test:
        test_train = 'test'
    else:
        test_train = 'train'
    process_data(test_train)


if __name__ == "__main__":
    main()
