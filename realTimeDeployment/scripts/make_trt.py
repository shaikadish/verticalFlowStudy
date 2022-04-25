"""Conversion of model to tensor rt for the Jetson Nano"""

# Imports
import torch
import torch.nn as nn
from torch.serialization import save
from torch2trt import torch2trt, TRTModule
from resnet_feature_output import *
from torch2trt.torch2trt import *

# Frame classifier architecture
res = 101
if res == 18:
    framenet = resnet18(num_classes=7).cuda().half().eval()
elif res == 101:
    framenet = resnet101(num_classes=7).cuda().half().eval()

# Load in original model
orginal_framenet_path = 'real_time_deployment/data/pretrained_models/original_version/'
model_name = 'model_name.pth'
framenet.load_state_dict(
    torch.load(
        orginal_framenet_path +
        model_name,
        map_location=torch.device('cuda')))

# Example input shape
x = torch.ones((1, 3, 256, 256)).half().cuda()

# Convert model
model_trt = torch2trt(framenet.half().eval().cuda(), [x])

# Save converted model
save_path = 'real_time_deployment/data/pretrained_models/trt_version/'
torch.save(model_trt.state_dict(), save_path +
           model_name[:-4] + '_trt' + '.pth')
