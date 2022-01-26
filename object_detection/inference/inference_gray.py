import numpy as np
import cv2
import matplotlib.pyplot as plt
from object_detection.backbones.ssd_resnet20 import SSD_ResNet20
from object_detection.backbones.ssd_mobilenet import SSD_MobileNet
from object_detection.backbones.ssd_minixception import SSD_MiniXception
from object_detection.backbones.ssd_squeezenet import SSD_SqueezeNet
from object_detection.backbones.ssd_densenet121 import SSD_DenseNet121
from object_detection.backbones.ssd_autoencoder import SSD_Autoencoder
from object_detection.dataloader.marine import label_map
from object_detection.trainer.pipelines_gray import DetectSingleShotGray
from object_detection.trainer.pipelines_gray import load_image_gray
from paz import processors as pr

model_bb = 'autoencoder'
# 604, 1777, 1815, 151, 1609
input_file = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation/Images/'
input_base_file = 'marine-debris-aris3k-1609.png'
input_file += input_base_file

image = load_image_gray(input_file)
print(image.shape)

ssd_resnet_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_resnet20.hdf5'
ssd_mobilenet_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_mobilenet.hdf5'
ssd_minixception_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_minixception.hdf5'
ssd_squeezenet_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_squeezenet.hdf5'
ssd_densenet121_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_densenet121.hdf5'
ssd_autoencoder_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_autoencoder.hdf5'

if model_bb == 'resnet20':
    # 1606, 54, 1470, 1628, 799
    weight_path = ssd_resnet_weights
    model = SSD_ResNet20(12, weight_folder='weights/')
elif model_bb == 'mobilenet':
    # 292, 894, 1655, 1808, 596
    weight_path = ssd_mobilenet_weights
    model = SSD_MobileNet(12, weight_folder='weights/')
elif model_bb == 'minixception':
    # 26, 931, 1372, 1637, 526
    weight_path = ssd_minixception_weights
    model = SSD_MiniXception(12, weight_folder='weights/')
elif model_bb == 'squeezenet':
    # 1181, 293, 170, 1690, 900
    weight_path = ssd_squeezenet_weights
    model = SSD_SqueezeNet(12, weight_folder='weights/')
elif model_bb == 'densenet121':
    # 441, 1275, 1405, 1799, 99
    weight_path = ssd_densenet121_weights
    model = SSD_DenseNet121(12, weight_folder='weights/')
elif model_bb == 'autoencoder':
    # 604, 1777, 1815, 151, 1609
    weight_path = ssd_autoencoder_weights
    model = SSD_Autoencoder(12, weight_folder='weights/')
else:
    weight_path = None
    model = None


model.load_weights(weight_path)
model.summary()

class_names = list(label_map.values())
pipeline = DetectSingleShotGray(model, class_names, 0.45, 0.45)
out = pipeline(image)
print(out['boxes2D'], out['image'].shape)
draw_process = pr.DrawBoxes2D(class_names)
image_drawn = draw_process(out['image'], out['boxes2D'])
filename = model_bb + '_' + input_base_file.split('.')[0].split('-')[-1] + '.jpg'
plt.imsave(filename, image_drawn)
