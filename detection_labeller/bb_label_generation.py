import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from create_xml_annotations import create_labimg_xml

label_map = {
    0:  'Background',
    1:  'Bottle',
    2:  'Can',
    3:  'Chain',
    4:  'Drink-carton',
    5:  'Hook',
    6:  'Propeller',
    7:  'Shampoo-bottle',
    8:  'Standing-bottle',
    9:  'Tire',
    10: 'Valve',
    11: 'Wall',
}

save_annotation_path = 'md_fls_dataset/data/watertank-segmentation/Annotations'

images_path = 'md_fls_dataset/data/watertank-segmentation/Images'
masks_path = 'md_fls_dataset/data/watertank-segmentation/Masks'

images_files = os.listdir(images_path)
masks_files = os.listdir(images_path)

for filename in images_files:
    sonar_image_path = os.path.join(images_path, filename)
    sonar_image = cv2.imread(sonar_image_path, cv2.IMREAD_GRAYSCALE)
    mask_filename = os.path.join(masks_path, filename)
    if not mask_filename:
        continue
    mask_image = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    print('\n')
    print("Loaded {}, image: {}, mask: {}".format(filename, sonar_image.shape,
                                                  mask_image.shape,))
    print("Image unique vals {}".format(np.unique(mask_image)))

    label_image = label(mask_image)
    props = regionprops(label_image)

    # fig, ax = plt.subplots(ncols=2, nrows=1)
    # out_image = sonar_image.copy()
    annotation_list = []
    label_list = []

    for region in props:
        # cv2.rectangle(out_image, (region.bbox[1], region.bbox[0]),
        #               (region.bbox[3], region.bbox[2]), (255, 0, 0), 2)
        y_min, x_min, y_max, x_max = region.bbox
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        print('Corner: ', region.bbox)
        print('Center: ', x, y, w, h)
        mask_label = int(np.max(mask_image[y_min:y_max, x_min:x_max]))
        print("Found label: ", mask_label, type(mask_label))
        annotation_list.append([x, y, w, h])
        label_list.append(label_map[mask_label])
    #     ax[0].imshow(label_image)
    #     ax[1].imshow(out_image)
    # plt.show()

    success = create_labimg_xml(sonar_image_path, annotation_list, label_list,
                                sonar_image.shape, filename,
                                save_annotation_path)
    if not success:
        raise ValueError('Error creating annotation')
