import os
import xml.etree.ElementTree as ET
import tensorflow as tf
from keras_cv import bounding_box

def parse_annotation(xml_file, class_mapping, path_images, single_cls: bool = False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    if not single_cls:
        class_ids = [
            list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
            for cls in classes
        ]
    else:
        class_ids = [0] * len(classes)
    return image_path, boxes, class_ids

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32),
            "bounding_boxes": bounding_boxes}
            #"img_name": os.path.basename(image_path)}

def dict_to_tuple(inputs):
    # https://github.com/keras-team/keras-io/issues/1475#issuecomment-1716460270
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=1000
    )
