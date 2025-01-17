import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


import numpy as np
from paz.models import SSD300
from paz.abstract import Processor, SequentialProcessor
from paz import processors as pr
from object_detection.trainer.pipelines import AugmentDetection
from object_detection.dataloader.marine import Marine
from object_detection.dataloader.marine import label_map


class ShowBoxes(Processor):
    def __init__(self, class_names, prior_boxes,
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(ShowBoxes, self).__init__()
        self.deprocess_boxes = SequentialProcessor([
            pr.DecodeBoxes(prior_boxes, variances),
            pr.ToBoxes2D(class_names, True),
            pr.FilterClassBoxes2D(class_names[1:])])
        self.denormalize_boxes2D = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(class_names)
        self.show_image = pr.ShowImage()
        self.resize_image = pr.ResizeImage((600, 600))

    def call(self, image, boxes):
        image = self.resize_image(image)
        boxes2D = self.deprocess_boxes(boxes)
        boxes2D = self.denormalize_boxes2D(image, boxes2D)
        image = self.draw_boxes2D(image, boxes2D)
        image = (image*255).astype(np.uint8)
        image = image[..., ::-1]
        self.show_image(image)
        return image, boxes2D


size = 300
split = 'train'
epochs = 1
batch_size = 30

ds_path = '/media/deepan/externaldrive1/project_repos/marine_od/'
ds_path += 'marine-debris-fls-datasets/md_fls_dataset/'
ds_path += 'data/watertank-segmentation'
Dataset = Marine(ds_path, label_map, 'train')
train_data = Dataset.load_data('train')
val_data = Dataset.load_data('val')
test_data = Dataset.load_data('test')

data = train_data[:1]

class_names = Dataset.class_names
model = SSD300(head_weights=None, base_weights=None, num_classes=12)
prior_boxes = model.prior_boxes

testor_encoder = AugmentDetection(prior_boxes, num_classes=12)
testor_decoder = ShowBoxes(class_names, prior_boxes)
sample_arg = 0
for sample_arg in range(1):
    sample = data[sample_arg]
    print(sample)
    print(sample)
    wrapped_outputs = testor_encoder(sample)
    print(wrapped_outputs['labels'].keys())
    image = wrapped_outputs['inputs']['image']
    boxes = wrapped_outputs['labels']['boxes']
    image, boxes = testor_decoder(image, boxes)
