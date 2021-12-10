from object_detection.dataloader.marine import Marine
from object_detection.dataloader.marine import label_map


ds_path = '/media/deepan/externaldrive1/project_repos/marine_od/'
ds_path += 'marine-debris-fls-datasets/md_fls_dataset/'
ds_path += 'data/watertank-segmentation'
Dataset = Marine(ds_path, label_map, 'train')
train_data = Dataset.load_data('train')
val_data = Dataset.load_data('val')
test_data = Dataset.load_data('test')