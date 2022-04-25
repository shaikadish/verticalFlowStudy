# verticalFlowStudy

The code found here was developed during a computer vision based study of Vertical CO2 flow, performed by the Atlas Experiment group, based at CERN. The structure of the repository is as follows:
```
├── FlowNet
│   ├── learning
│   ├── scripts
│   └── utils.py
├── FrameNet
│   ├── learning
│   └── utils.py
├── VaporNet
│   ├── learning
│   ├── scripts
│   └── utils.py
├── breakpoint_algorithm
│   ├── functions
│   └── refraction_correction
└── real_time_deployment
    ├── real_time_interpolation
    ├── scripts
    └── utils.py
```

These sub-directories can be further explained as follows:

## FlowNet, FrameNet and VaporNet

The code found in these subdirectories was used in the developement of a *CNN + LSTM* neural network model used for the classification of flow images (```FrameNet```), the classification of flow regime from a video clip (```FlowNet```), and the estimation of the flow's vapor quality from a video clip (```VaporNet```). This model, and the relationships between its three components is further described in [this published work](https://www.mdpi.com/1424-8220/22/3/996). 
Each of these sub-directories has the following items:

- ```learning```: A folder containing the code used for training and testing each respective components of the model. This include each component's architecture (```model.py```) and scripts to train and evaluate the component (```train.py``` and ```test.py``` respectively). These Python scripts make use of the PyTorch library.
- ```scripts```: This folder contains scripts used for pre-processing inputs for the *LSTM* networks used by ```VaporNet``` and ```FrameNet```. This pre-processing converts input videos to a set of a sequences of high level image features in the form of vector encodings, which can be used for the training of *LSTN* networks.
- ```utils.py```: This Python script contains utility functions utilized during the training, testing and pre-processing required for each model component.

The code found within these sub-directories includes further useful documentation.

## real_time_deployment

Found here are the scripts to deploy FlowNet, FrameNet and VaporNet in real-time using a *Jetson Nano* micro-computer , as was done in [a related publication](https://www.mdpi.com/1424-8220/22/3/996).
The items in this sub-directory are as follows:

- ```real_time_interpolation```: Scipts to run VaporNet and FrameNet in real-time.
- ```scripts```: Helper scripts used to support real-time deployment. Found here are ```camera_test.py```, used to allign the *Jetson Nano's* camera correctly with the tube section being observed, and ```make_trt.py```, a script used to convert an existing PyTorch model into ```trt``` format, so as to optimize it for deployment on the Nano.
- ```utils.py```: Utility functions used in the real-time deployment of these models.

The code found within these sub-directories includes further useful documentation.

## breakpoint_algorithm

This sub-directory includes MatLab code used to detect the outline of bubble's from images of bubbly flow. This algorithm was used to detect the bubbles within a frame in order to calculate the void fraction observed within that frame. This work is detailed in my Master's thesis, which will be linked once finalised.
The folders in this sub-directories are as follows:

- ```functions```: The MatLab functions used to detect bubbles with an image. ```bp_algorithm.m``` is the main function, with the other functions acting as helper functions.
- ```refraction_correction```: Found here are pre-processing steps to correct for the distortion caused by a the glass tube when observing a section of the flow-channel.

The code found within these sub-directories includes further useful documentation.
