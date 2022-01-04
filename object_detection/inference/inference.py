import numpy as np
import cv2
import matplotlib.pyplot as plt
from object_detection.backbones.ssd300 import SSD300
from object_detection.dataloader.marine import label_map
from object_detection.trainer.pipelines import DetectSingleShotGray

input_file = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation/Images/'
input_file += 'marine-debris-aris3k-336.png'

image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
image = np.stack((image,)*3, axis=-1)
print(image.shape)

weight_path = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/trained_models/SSD300/weights.hdf5'

model = SSD300(12, None, None)
model.load_weights(weight_path)
model.summary()

class_names = list(label_map.values())
pipeline = DetectSingleShotGray(model, class_names, 0.45, 0.45)
out = pipeline(image)
print(out['boxes2D'])
plt.imsave('image.jpg', out['image'])
