import os
from pathlib import Path
import xml.etree.cElementTree as ET
from xml.dom import minidom


def create_labimg_xml(image_path, annotation_list, label_list, image_shape,
                      filename, save_annotation_path):
    image_path = image_path.replace(
        'md_fls_dataset/data/watertank-segmentation/', '')
    image_path = Path(image_path)
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(image_path.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path.name)
    ET.SubElement(annotation, 'path').text = str(image_path)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Forward-Looking Sonar Marine Debris Datasets'
    ET.SubElement(source, 'annotation').text = 'COCO Format - x y w h'
    ET.SubElement(source, 'images').text = 'Watertank Dataset'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image_shape[1])
    ET.SubElement(size, 'height').text = str(image_shape[0])
    ET.SubElement(size, 'depth').text = str(1)

    ET.SubElement(annotation, 'segmented').text = '1'

    for coordinate, label in zip(annotation_list, label_list):
        xmin, ymin = coordinate[0], coordinate[1]
        w, h = coordinate[2], coordinate[3]

        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = str(label)
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'x').text = str(xmin)
        ET.SubElement(bndbox, 'y').text = str(ymin)
        ET.SubElement(bndbox, 'w').text = str(w)
        ET.SubElement(bndbox, 'h').text = str(h)

    annotation_str = minidom.parseString(
        ET.tostring(annotation)).toprettyxml(indent="\t")
    xml_file_name = os.path.join(save_annotation_path,
                                 filename.split('.png')[0] + '.xml')
    f = open(xml_file_name, "w+")
    f.write(annotation_str)
    print("Created XML annotation: ", xml_file_name)

    return True
