# MM-EgoGesture
Ego-centric hand gesture recognition with multi-modal sensors.

## Environment
* Python: 3.10.8
* Pytorch: 1.13.1
* CUDA: 11.6
* CuDNN: 8
* Environment can directly be imported through [Docker image](https://hub.docker.com/repository/docker/gogoho88/stanford_mmwave/tags/v3/sha256-481efb7f0500f3657296cd8e1320404887e18f49a2e6683fbcec18d6a9e7d212)

Using the `mmegogesture.yaml`
'''
conda env create -f env.yaml
'''

## Preparing the Dataset
* Download the dataset from this [Google Drive link](https://drive.google.com/drive/folders/1sXZ0JFAW5JQ_2f_Y19F3As97P0vvQ913?usp=share_link)
* Unzip folders `Data (Cleaned)` and `Metadata`
* Data has all the samples with structure of
(named as `{episode}-{idx(1~12)}-{modality}.npy`)
```
dataset
  ├── 20231208153042-1-cam.npy
  ├── 20231208153042-1-rad-uD.npy
  ...
```

## Argument configurations
This codebase uses [Hydra](https://github.com/facebookresearch/hydra) to manage and configure arguments. Hydra offers more flexibility for running and managing complex configurations, and supports rich hierarchical config structures.

The YAML configuration files are in folder `conf`. So you can have a set of arguments in your YAML file like
```
train:
  learning_rate: 1e-4
sensor: 
  select: ['rad-uD', 'cam-img']
```

## Code Tree
`/main_gesture.py`: Main file to run the code<br>
`/conf/`: Configuration file for adjusting parameters<br>
`/model/`: Include 2D/3D neural network models<br>
`/utils/`: Utility functions for training and testing models such as dataloader, data transformation, model training/testing 

## Train the Baseline Model
* Select single- or multi-sensor input modality in `/conf/config_gesture.yaml`
* Specify data folder and metadata file through `path_data` and `path_des` in `/conf/config_gesture.yaml`
* Rut it through python
```
cd MM-EgoGesture
python main_gesture.py
```

## Model Inference and Checkpoints
Trained model parameters(`model_Fusion.pt`) and corresponding settings(`args_Fusion.yaml`) can be found from this [Google Drive foler](https://drive.google.com/drive/folders/1n1nkfOQtNZ2cDRLQldMPpkFfIqnB4RXG?usp=share_link).

To directly do inference using the trained model, you need to change the config file:
* In `/conf/config_inference.yaml`, specify the path of `model_Fusion.pt` and `args_Fusion.yaml` with `path_model` and `path_args`, respectively.
* Specify the path where the result will be saved in `path_save`
* Run it through Python
```
cd MM-EgoGesture
python main_inference.py
```

## License
The `MM-EgoGesture` dataset is published under the CC BY-NC-ND License, and all codes are published under the Apache License 2.0.