import numpy as np
import cv2
import matplotlib.pyplot as plt
from object_detection.backbones.ssd_resnet20 import SSD_ResNet20
from object_detection.backbones.ssd_mobilenet import SSD_MobileNet
from object_detection.dataloader.marine import label_map
from object_detection.trainer.pipelines_gray import DetectSingleShotGray
from object_detection.trainer.pipelines_gray import load_image_gray
from paz import processors as pr

model_bb = 'resnet20'
input_file = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation/Images/'
input_file += 'marine-debris-aris3k-54.png'

image = load_image_gray(input_file)
print(image.shape)

ssd_mobilenet_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_mobilenet.hdf5'
ssd_resnet_weights = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/weights/marine_debris_ssd_resnet20.hdf5'

if model_bb == 'resnet20':
    # 1606, 54
    weight_path = ssd_resnet_weights
    model = SSD_ResNet20(12, weight_folder='weights/')
else:
    # 292, 894
    weight_path = ssd_mobilenet_weights
    model = SSD_MobileNet(12, weight_folder='weights/')

model.load_weights(weight_path)
model.summary()

class_names = list(label_map.values())
pipeline = DetectSingleShotGray(model, class_names, 0.45, 0.45)
out = pipeline(image)
print(out['boxes2D'], out['image'].shape)
draw_process = pr.DrawBoxes2D(class_names)
image_drawn = draw_process(out['image'], out['boxes2D'])
plt.imsave('image.jpg', image_drawn)
