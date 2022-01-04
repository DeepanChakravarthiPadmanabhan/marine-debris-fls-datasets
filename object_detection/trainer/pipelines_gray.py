import os
import cv2
import numpy as np
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor
from tensorflow.keras.callbacks import Callback
from paz.evaluation import evaluateMAP
from paz.backend.boxes import compute_iou


class StackImages(Processor):
    """Converts grayscale image to 3 channel input for SSD.

    # Arguments
        image: Image of shape (height, width)

    """

    def __init__(self):
        super(StackImages, self).__init__()

    def call(self, image):
        image = np.stack((image,) * 3, axis=-1)
        return image


class LoadImageGray(Processor):
    """Loads image.

    # Arguments
        num_channels: Integer, valid integers are: 1, 3 and 4.
    """
    def __init__(self):
        super(LoadImageGray, self).__init__()

    def call(self, image):
        return load_image_gray(image)


def load_image_gray(filepath):
    """Load image from a ''filepath''.

    # Arguments
        filepath: String indicating full path to the image.
        num_channels: Int.

    # Returns
        Numpy array.
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = np.expand_dims(image, -1)
    return image


class NormalizeImageGray(Processor):
    """Normalize image by diving all values by 255.0.
    """
    def __init__(self):
        super(NormalizeImageGray, self).__init__()

    def call(self, image):
        return image - 84.51


class ExpandDims(Processor):
    """Loads image.

    # Arguments
        num_channels: Integer, valid integers are: 1, 3 and 4.
    """
    def __init__(self):
        super(ExpandDims, self).__init__()

    def call(self, image):
        return np.expand_dims(image, -1)


class RandomSampleCrop(Processor):
    """Crops and image while adjusting the bounding boxes.
    Boxes should be in point form.
    # Arguments
        probability: Float between ''[0, 1]''.
    """
    def __init__(self, probability=0.5):
        self.probability = probability
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        super(RandomSampleCrop, self).__init__()

    def call(self, image, boxes):
        if self.probability < np.random.rand():
            return image, boxes
        labels = boxes[:, -1:]
        boxes = boxes[:, :4]
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                boxes = np.hstack([boxes, labels])
                return image, boxes

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = compute_iou(rect, boxes)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2]]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                return current_image, np.hstack(
                    [current_boxes, current_labels])


def flip_left_right(boxes, width):
    """Flips box coordinates from left-to-right and vice-versa.
    # Arguments
        boxes: Numpy array of shape `[num_boxes, 4]`.
    # Returns
        Numpy array of shape `[num_boxes, 4]`.
    """
    flipped_boxes = boxes.copy()
    flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return flipped_boxes


class RandomFlipBoxesLeftRight(Processor):
    """Flips image and implemented labels horizontally.
    """
    def __init__(self):
        super(RandomFlipBoxesLeftRight, self).__init__()

    def call(self, image, boxes):
        if np.random.randint(0, 2):
            boxes = flip_left_right(boxes, image.shape[1])
            image = image[:, ::-1]
        return image, boxes


class Expand(Processor):
    """Expand image size up to 2x, 3x, 4x and fill values with mean color.
    This transformation is applied with a probability of 50%.

    # Arguments
        max_ratio: Float.
        mean: None/List: If `None` expanded image is filled with
            the image mean.
        probability: Float between ''[0, 1]''.
    """
    def __init__(self, max_ratio=2, mean=None, probability=0.5):
        super(Expand, self).__init__()
        self.max_ratio = max_ratio
        self.mean = mean
        self.probability = probability

    def call(self, image, boxes):
        if self.probability < np.random.rand():
            return image, boxes
        image = np.stack((image[:, :, 0],) * 3, -1)
        height, width, num_channels = image.shape
        ratio = np.random.uniform(1, self.max_ratio)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)
        expanded_image = np.zeros((int(height * ratio),
                                   int(width * ratio), num_channels),
                                  dtype=image.dtype)

        if self.mean is None:
            expanded_image[:, :, :] = np.mean(image, axis=(0, 1))
        else:
            expanded_image[:, :, :] = self.mean

        expanded_image[int(top):int(top + height),
                       int(left):int(left + width)] = image
        expanded_boxes = boxes.copy()
        expanded_boxes[:, 0:2] = boxes[:, 0:2] + (int(left), int(top))
        expanded_boxes[:, 2:4] = boxes[:, 2:4] + (int(left), int(top))
        expanded_image = expanded_image[:, :, 0]
        expanded_image = np.expand_dims(expanded_image, -1)
        return expanded_image, expanded_boxes


class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.
    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToImageBoxCoordinates())
        self.add(Expand(mean=mean))
        self.add(RandomSampleCrop())
        self.add(RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes
    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes, IOU),)
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class PreprocessImage(SequentialProcessor):
    """Preprocess RGB image by resizing it to the given ``shape``. If a
    ``mean`` is given it is substracted from image and it not the image gets
    normalized.
    # Arguments
        shape: List of two Ints.
        mean: List of three Ints indicating the per-channel mean to be
            subtracted.
    """
    def __init__(self, shape, mean=None):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImage(float))
        self.add(NormalizeImageGray())
        self.add(ExpandDims())


class AugmentImage(SequentialProcessor):
    """Augments an RGB image by randomly changing contrast, brightness
        saturation and hue.
    """
    def __init__(self):
        super(AugmentImage, self).__init__()
        self.add(pr.RandomContrast())
        self.add(pr.RandomBrightness())


class AugmentDetection(SequentialProcessor):
    """Augment boxes and images for object detection.
    # Arguments
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, prior_boxes, split=pr.TRAIN, num_classes=12, size=300,
                 mean=None, IOU=.5, variances=[0.1, 0.1, 0.2, 0.2]):
        super(AugmentDetection, self).__init__()
        # image processors
        self.augment_image = AugmentImage()
        self.preprocess_image = PreprocessImage((size, size), mean)

        # box processors
        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes']))
        self.add(pr.ControlMap(LoadImageGray(), [0], [0]))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 1]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


class DetectSingleShotGray(Processor):
    """Single-shot object detection prediction.
    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating the per channel mean.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=None, variances=[0.1, 0.1, 0.2, 0.2],
                 draw=False):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw

        super(DetectSingleShotGray, self).__init__()
        preprocessing = SequentialProcessor(
            [pr.ResizeImage(self.model.input_shape[1:3]),
             NormalizeImageGray(),
             pr.CastImage(float),
             pr.ExpandDims(axis=0),
             pr.ExpandDims(axis=-1)])
        postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(self.model.prior_boxes, self.variances),
             pr.NonMaximumSuppressionPerClass(self.nms_thresh),
             pr.FilterBoxes(self.class_names, self.score_thresh)])
        self.predict = pr.Predict(self.model, preprocessing, postprocessing)

        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        image = image[:, :, 0]
        boxes2D = self.predict(image)
        boxes2D = self.denormalize(image, boxes2D)
        image = np.stack((image,) * 3, axis=-1)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class EvaluateMAPGray(Callback):
    """Evaluates mean average precision (MAP) of an object detector.
    # Arguments
        data_manager: Data manager and loader class. See ''paz.datasets''
            for examples.
        detector: Tensorflow-Keras model.
        period: Int. Indicates how often the evaluation is performed.
        save_path: Str.
        iou_thresh: Float.
    """
    def __init__(
            self, dataset, detector, period, save_path, iou_thresh=0.5,
            class_names=None):
        super(EvaluateMAPGray, self).__init__()
        self.detector = detector
        self.period = period
        self.save_path = save_path
        self.dataset = dataset
        self.iou_thresh = iou_thresh
        self.class_names = class_names
        self.class_dict = {}
        for class_arg, class_name in enumerate(self.class_names):
            self.class_dict[class_name] = class_arg

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.period == 0:
            result = evaluateMAP(
                self.detector,
                self.dataset,
                self.class_dict,
                iou_thresh=self.iou_thresh,
                use_07_metric=True)

            result_str = 'mAP: {:.4f}\n'.format(result['map'])
            metrics = {'mAP': result['map']}
            for arg, ap in enumerate(result['ap']):
                if arg == 0 or np.isnan(ap):  # skip background
                    continue
                metrics[self.class_names[arg]] = ap
                result_str += '{:<16}: {:.4f}\n'.format(
                    self.class_names[arg], ap)
            print(result_str)

            # Saving the evaluation results
            filename = os.path.join(self.save_path, 'MAP_Evaluation_Log.txt')
            with open(filename, 'a') as eval_log_file:
                eval_log_file.write('Epoch: {}\n{}\n'.format(
                    str(epoch), result_str))
