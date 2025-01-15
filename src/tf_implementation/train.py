import json
import os
from datetime import datetime

from picsellia.types.enums import LogType
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow import keras

import keras_cv

from src.tf_implementation.dataset import parse_annotation, load_dataset, dict_to_tuple
from src.tf_implementation.loggers.picsellia_logger import get_picsellia_experiment, log_split_table
from src.tf_implementation.metrics import EvaluateCOCOMetricsCallback
from src.tf_implementation.utils import count_annotations_in_split

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 2
GLOBAL_CLIPNORM = 10.0
IMAGE_SIZE = (640, 640)

if __name__ == "__main__":

    class_ids = [
        "Model",
        "Added",
    ]
    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    # Path to images and annotations
    path_images = "../../dataset_pascal_voc/Evry_dataset_2024_pascal_voc/VOCdevkit/VOC/Samples_test/JPEGImages"
    path_annot = "../../dataset_pascal_voc/Evry_dataset_2024_pascal_voc/VOCdevkit/VOC/Samples_test/Annotations"

    # Get all XML file paths in path_annot and sort them
    xml_files = sorted(
        [
            os.path.join(path_annot, file_name)
            for file_name in os.listdir(path_annot)
            if file_name.endswith(".xml")
        ]
    )

    # Get all JPEG image file paths in path_images and sort them
    jpg_files = sorted(
        [
            os.path.join(path_images, file_name)
            for file_name in os.listdir(path_images)
            if file_name.endswith(".jpg")
        ]
    )

    image_paths = []
    bbox = []
    classes = []
    for xml_file in tqdm(xml_files):
        image_path, boxes, class_ids = parse_annotation(xml_file, class_mapping, path_images, single_cls=True)
        image_paths.append(image_path)
        bbox.append(boxes)
        classes.append(class_ids)

    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)

    data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

    # Determine the number of validation samples
    num_val = int(len(xml_files) * SPLIT_RATIO)

    # Split the dataset into train and validation sets
    val_data = data.take(num_val)
    train_data = data.skip(num_val)

    ######## data augmentation #########

    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal_and_vertical", bounding_box_format="xyxy"),
            keras_cv.layers.RandomShear(
                x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
            ),
            keras_cv.layers.JitteredResize(
                target_size=IMAGE_SIZE, scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
            ),
            # layers.RandomZoom(0.2),
            # keras_cv.layers.RandomRotation(0.2),
            # keras.layers.RandomBrightness(factor=0.2),

        ]
    )

    train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    resizing = keras_cv.layers.JitteredResize(
        target_size=IMAGE_SIZE,
        scale_factor=(0.75, 1.3),
        bounding_box_format="xyxy",
    )

    val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    ######### Model #########
    # backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    #     "yolo_v8_xs_backbone_coco"  # We will use yolov8 small backbone with coco weights
    # )

    # https://www.kaggle.com/api/v1/models/keras/yolov8/keras/yolo_v8_xs_backbone_coco/2/download/
    xs_configuration_yolo_folder = r"C:\Users\tristan_cotte\Downloads\yolov8-keras-yolo_v8_xs_backbone_coco-v2.tar\yolov8-keras-yolo_v8_xs_backbone_coco-v2"
    config_file = os.path.join(xs_configuration_yolo_folder, "config.json")

    with open(config_file, 'r') as JSON:
        json_dict = json.load(JSON)

    backbone = keras_cv.models.YOLOV8Backbone.from_config(json_dict['config'])

    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )

    yolo.compile(
        optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou", run_eagerly=True,
        jit_compile=False
    )

    date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_dir = os.path.join('experiments', f'run_{date_time}')
    os.makedirs(run_dir, exist_ok=True)
    config_file = os.path.join(run_dir, "configs.txt")

    with open(config_file, "w") as file:
        file.write(f"""
        SPLIT_RATIO = {SPLIT_RATIO}
        BATCH_SIZE = {BATCH_SIZE}
        LEARNING_RATE = {LEARNING_RATE}
        EPOCHS = {EPOCHS}
        GLOBAL_CLIPNORM = {GLOBAL_CLIPNORM}
        IMAGE_SIZE = {IMAGE_SIZE}
        """)

    train_val_split_elements = {"train": len(train_data),
                                "val": len(val_data)}

    ann_in_train_split = count_annotations_in_split(split=train_data, class_mapping=class_mapping)
    ann_in_val_split = count_annotations_in_split(split=val_data, class_mapping=class_mapping)

    picsellia_experiment = get_picsellia_experiment()
    log_split_table(picsellia_experiment=picsellia_experiment, annotations_in_split=train_val_split_elements,
                    title="Nb elem / split")
    log_split_table(picsellia_experiment=picsellia_experiment, annotations_in_split=ann_in_train_split,
                    title="Train split")
    log_split_table(picsellia_experiment=picsellia_experiment, annotations_in_split=ann_in_val_split,
                    title="Validation split")

    data = {
        'split_ratio': SPLIT_RATIO,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "global_clipnorm": GLOBAL_CLIPNORM,
        "image_size": IMAGE_SIZE
    }
    picsellia_experiment.log(name='Parameters', type=LogType.TABLE, data=data)
    picsellia_experiment.log(name='LabelMap', type=LogType.TABLE,
                             data={str(key): value for key, value in class_mapping.items()})

    yolo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[EvaluateCOCOMetricsCallback(val_ds, "../models/model.h5",
                                               picsellia_experiment=picsellia_experiment,
                                               config_file=config_file)],
    )
