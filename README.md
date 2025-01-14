# Forward-Looking Sonar Marine Debris Datasets
Marine-Debris Datasets captured with a Forward-Looking Sonar in a water tank and turntable using an ARIS Explorer 3000 Sonar.

# SSD trained weights

The weights of SSD trained with different backbones such as VGG16, ResNet20, MobileNet, DenseNet121, SqueezeNet, and MiniXception are available [here](https://github.com/DeepanChakravarthiPadmanabhan/model-weights/releases/tag/v2.0.0).

# Watertank Dataset

The full dataset release containing marine debris in the OSL water tank can be found [here](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0).

## Matching Task

This tasks consists of learning to match two sonar image patches. The subdataset release as HDF5 files is available [here](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-match-v1.0).

## Classification Task

Coming soon.

## Segmentation Task

Per-pixel segmentation labels have been made by Deepak Singh from Netaji Subhas Institute Of Technology, and consists of 11 classes plus background. This task is available [here](https://github.com/mvaldenegro/marine-debris-fls-datasets/tree/master/md_fls_dataset/data/watertank-segmentation). A preprint about this dataset is coming soon.

# Turntable Dataset

The turntable dataset is captured with the sonar in a fixed position and pose, while objects are placed in a rotating turntable, allowing to capture a full yaw rotation of each object. These images are available [here](https://github.com/mvaldenegro/marine-debris-fls-datasets/tree/master/md_fls_dataset/data/turntable-cropped). For now only a classification task is available.

# Acknowledgements
This work has been partially supported by the FP7-PEOPLE-2013-ITN project ROBOCADEMY (Ref 608096) funded by the European Commission.
