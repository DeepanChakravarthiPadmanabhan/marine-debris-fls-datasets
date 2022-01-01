import os
import random
from xml.etree import ElementTree
import numpy as np
import cv2
import matplotlib.pyplot as plt

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


class Marine:
    def __init__(self, path, class_names, split='train'):
        self.path = path
        self.arg_to_class = class_names
        self.class_to_arg = {value: key for key, value
                             in class_names.items()}
        self.class_names = list(class_names.values())
        self.num_classes = len(class_names)
        image_files, annotation_files = self.get_filenames(path)
        (self.train_images, self.train_annotations, self.val_images,
         self.val_annotations, self.test_images, self.test_annotations
         ) = self.split_files(image_files, annotation_files)
        self.split = split
        if split == 'train':
            # Save the test image and use it for trained model
            np.savetxt('test.txt', np.array(self.test_images), fmt='%s')

    def get_filenames(self, path):
        images_path = os.path.join(path, 'Images')
        image_names = os.listdir(images_path)
        random.shuffle(image_names)
        image_files = []
        annotation_files = []
        for files in image_names:
            absolute_file = os.path.join(images_path, files)
            if absolute_file:
                image_files.append(absolute_file)
            path = absolute_file.replace('/Images/', '/Annotations/')
            filename = path.split('.png')[0] + '.xml'
            if os.path.exists(filename):
                annotation_files.append(filename)
        return image_files, annotation_files

    def split_files(self, image_files, annotation_files):
        total_images = len(image_files)
        train_split = int(0.7 * total_images)
        val_split = int(0.2 * total_images)
        train_annotations = annotation_files[:train_split]
        train_images = image_files[:train_split]
        val_annotations = annotation_files[train_split:
                                           train_split + val_split]
        val_images = image_files[train_split: train_split + val_split]
        test_annotations = annotation_files[train_split + val_split:]
        test_images = image_files[train_split + val_split:]
        return (train_images, train_annotations, val_images, val_annotations,
                test_images, test_annotations)

    def get_split_data(self, split_name):
        if split_name == 'train':
            return self.train_images, self.train_annotations
        elif split_name == 'val':
            return self.val_images, self.val_annotations
        elif split_name == 'test':
            images = np.loadtxt('test.txt', dtype='str')
            replacer = lambda x: x.replace('/Images/', '/Annotations/')
            ext_changer = lambda x: x.replace('.png', '.xml')
            replacer = np.vectorize(replacer)
            ext_changer = np.vectorize(ext_changer)
            annotations = replacer(images)
            annotations = ext_changer(annotations)
            return images, annotations
        else:
            raise ValueError('Give proper split name.')

    def load_data(self, split_name):
        images, annotations = self.get_split_data(split_name)
        data = []
        for idx in range(len(images)):
            tree = ElementTree.parse(annotations[idx])
            root = tree.getroot()
            size_tree = root.find('size')
            image_width = float(size_tree.find('width').text)
            image_height = float(size_tree.find('height').text)
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            if split_name == 'test':
                width, height = 1, 1
            box_data = []
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                if class_name in self.class_to_arg:
                    class_arg = self.class_to_arg[class_name]
                    if split_name != 'test':
                        class_arg = get_random_label(
                            class_arg, self.class_to_arg)
                    bounding_box = object_tree.find('bndbox')

                    x = float(bounding_box.find('x').text)
                    y = float(bounding_box.find('y').text)
                    w = float(bounding_box.find('w').text)
                    h = float(bounding_box.find('h').text)
                    xmin = x
                    ymin = y
                    xmax = xmin + w
                    ymax = ymin + h
                    if split_name != 'test':
                        xmin, ymin, xmax, ymax = get_random_box(
                            xmin, ymin, xmax, ymax, image_height, image_width)

                    xmin, xmax = xmin / width, xmax / width
                    ymin, ymax = ymin / height, ymax / height

                    box_data.append([xmin, ymin, xmax, ymax, class_arg])

            data.append({'image': images[idx], 'boxes': np.asarray(box_data),
                         'image_index': images[idx]})

        return data


def check_box(image_path, boxes):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    fig, ax = plt.subplots(ncols=1, nrows=1)
    for box in boxes:
        image = cv2.rectangle(image, (int(box[1] * h), int(box[0] * w)),
                              (int(box[3] * h), int(box[2] * w)),
                              (255, 0, 0), 2)
    ax.imshow(image)
    plt.show()


def get_random_label(class_name, class_to_arg):
    class_out = class_name
    while class_out == class_name:
        class_out = random.randint(0, len(class_to_arg)-1)
    return class_out


def get_random_box(xmin, ymin, xmax, ymax, image_height, image_width):
    bb1 = {'x1': xmin, 'x2': xmax, 'y1': ymin, 'y2': ymax}
    x1, x2, y1, y2 = -1, -2, -1, -2
    iou = 0
    while iou == 0 and x1 != x2 and y1 != y2:
        x1 = random.randint(0, image_width-5)
        x2 = random.randint(x1+1, image_width)
        y1 = random.randint(0, image_height-5)
        y2 = random.randint(y1+1, image_height)
        bb2 = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
        iou = get_iou(bb1, bb2)
    return float(x1), float(y1), float(x2), float(y2)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
