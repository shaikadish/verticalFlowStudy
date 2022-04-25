import argparse
import torch
import torchvision.transforms as transforms
from resnet_feature_output import *
import cv2
from torch2trt import torch2trt, TRTModule
import time
from VaporNet.learning.model import LSTM_regression as LSTM
import warnings
import datetime
from utils import ArducamUtils
warnings.filterwarnings("ignore")

# Makes use of CUDA enabled GPU if one is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Class which executes program script in its construction
class classify_frames():
    """
    A class used to classify frames using FlowNet in real time

    Attributes
    ----------
    mf : float
        steady state mass flow rate of flow under observation
    tsat : float
        saturation temperature of flow under investigation
    LSTM_num_layers : int
        Sets the number of layers of the deep LSTM
    LSTM_hidden_size : int
        Sets the size of the hidden state of the LSTM
    LSTM_input_size : int
        Sets the number of input features to the LSTM
    LSTM_num_classes : int
        Number of classes being classified
    LSTM_seq_length : int
        length of feature sequeunce
    framenet : Object
        FrameNet model used for feature extraction
    vapornet : Object
        VaporNet LSTM network used for vapor quality regression
    transform : dict
        image transforms used during image preprocessing

    Methods
    -------
    live_interpolation()
        Function used for live vapor quality estimation.
        This function also saves all predictions.
    """

    def __init__(self):

        # Parse input argument(resnet version)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mf",
            help="resenet architecture used",
            type=float,
            default=10)
        parser.add_argument(
            "--tsat",
            help="resenet architecture used",
            type=float,
            default=10)
        self.mf = parser.mf
        self.tsat = parser.tsat

        # Model Paths
        path_framenet = 'real_time_deployment/data/pretrained_models/FrameNet/trt_version'
        path_vapornet = 'real_time_deployment/data/pretrained_models/VaporNet/model_name.pth'

        # LSTM configuration
        params_LSTM = path_vapornet.split('/')[-1]
        self.LSTM_seq_length = int(params_LSTM.split('_')[2])
        self.LSTM_input_size = 2048
        self.LSTM_hidden_size = int(params_LSTM.split('_')[3])
        self.LSTM_num_layers = int(params_LSTM.split('_')[4])
        self.LSTM_num_flat = int(params_LSTM.split('_')[5])
        self.LSTM_num_classes = 1
        vapornet = LSTM(
            self.LSTM_input_size,
            self.LSTM_hidden_size,
            self.LSTM_num_layers,
            self.LSTM_num_classes,
            self.LSTM_num_flat).half().to(device)
        vapornet.eval()

        # Load pretrained model parameters
        framenet = TRTModule()
        framenet.load_state_dict(
            torch.load(
                path_framenet,
                map_location=torch.device(device)))
        print('\n[*]Frame Network parameters loaded')
        vapornet.load_state_dict(
            torch.load(
                path_vapornet,
                map_location=torch.device(device)))
        print('\n[*]LSTM Network parameters loaded')
        self.framenet = framenet.eval()
        self.vapornet = vapornet.eval()

        # Image transforms for their input into FrameNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            # Centre crop with same aspect ratio as that used during training
            transforms.CenterCrop((252, 68)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Main script for real time data processing
        self.live_interpolation()

    # Extracts frames and frame classifications from input video

    @torch.no_grad()
    def live_interpolation(self):

        # Configure video stream
        capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        arducam_utils = ArducamUtils(0)
        capture.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils.convert2rgb)

        # Tensor of feature sequence produced by FrameNet for input into the
        # LSTM
        feature_input = torch.zeros((50, self.LSTM_input_size))

        # Ensures enough frames have been processed before interpolation begins
        loading_count = 0

        # Time checks
        FPS_time = time.time()
        data_time = time.time()

        # Sample rate paramaters [seconds]
        frame_rate = 1 / 10
        data_rate = 2

        # Accessing video stream
        while True:

            # Process current frame
            ret, image = capture.read()
            image = cv2.flip(image, 0)
            w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            image = image.reshape(int(h), int(w))
            image = arducam_utils.convert(image)

            if not ret:
                print('No Image Returned')
                break

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = image[:, 240:-240, :]
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

            # Only analyze a frame every <frame rate> seconds
            if time.time() - FPS_time >= frame_rate:
                FPS_time = time.time()

                # Frame feature extraction
                img = self.transform(image).half().cuda()
                features_frame, outputs_frame = self.framenet(
                    img.unsqueeze(dim=0))

                # Full up feature_input before processing begins
                if loading_count < 50:
                    feature_input[loading_count] = features_frame
                    loading_count += 1
                    if loading_count == 50:
                        LSTM_input = torch.tensor(
                            feature_input, dtype=torch.float16).reshape(-1, 1).reshape(-1, 50, self.LSTM_input_size)
                        LSTM_output = self.vapornet(torch.tensor(LSTM_input).half().cuda(), torch.tensor(
                            [self.tsat]).half().cuda(), torch.tensor([self.mf]).half().cuda())
                        loading_count += 1
                        print(f"Estimated Vapor Quality: {LSTM_output.item()}")

                # feature_input is full, and processing can begin
                else:
                    # Last-in-first-out shuffle of feature_input
                    feature_input[:-1, :] = feature_input[1:, :].clone()
                    feature_input[-1, :] = features_frame

                    # VaporNet output
                    LSTM_input = torch.tensor(feature_input,
                                              dtype=torch.float16).reshape(-1,
                                                                           1).reshape(-1,
                                                                                      50,
                                                                                      self.LSTM_input_size)
                    LSTM_output = self.vapornet(torch.tensor(LSTM_input).half().cuda(), torch.tensor(
                        [self.tsat]).half().cuda(), torch.tensor([self.mf]).half().cuda())
                    print(f"Estimated Vapor Quality: {LSTM_output.item()}")

                    # Only save a prediction every <data_rate> seconds
                    if(time.time() - data_time > data_rate):
                        data_time = time.time()
                        with open("vq_output.csv", "a", newline="") as writer:
                            # Saves the VaporNet ouput and the time at which it
                            # was taken
                            writer.write(
                                f"{datetime.datetime.now().isoformat(timespec='minutes')},{round(LSTM_output.item(),3)}\n")


def main():
    classify_frames()


if __name__ == '__main__':
    main()
