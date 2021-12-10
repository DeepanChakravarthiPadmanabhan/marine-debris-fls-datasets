import numpy as np
import cv2

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import regionprops, label

input_filename = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation/Masks/marine-debris-aris3k-264.png'
sonar_filename = '/media/deepan/externaldrive1/project_repos/marine_od/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation/Images/marine-debris-aris3k-264.png'

mask_image = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)
sonar_image = cv2.imread(sonar_filename, cv2.IMREAD_GRAYSCALE)

print("Loaded {} as {} {}".format(input_filename, mask_image.shape, mask_image.dtype))
print("Image unique vals {}".format(np.unique(mask_image)))

label_image = label(mask_image)
props = regionprops(label_image)

fig, ax = plt.subplots(ncols=2, nrows=1)

out_image = sonar_image

# TODO: Get correct class labels for each bounding box, from the original mask image
for region in props:
    cv2.rectangle(out_image, (region.bbox[1], region.bbox[0]), (region.bbox[3], region.bbox[2]), (255, 0, 0), 2)
    print(region.bbox)

ax[0].imshow(label_image, interpolation="nearest")
ax[1].imshow(out_image)
plt.show()
