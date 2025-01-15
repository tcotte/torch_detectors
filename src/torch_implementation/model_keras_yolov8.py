import os

import torch
from albumentations.pytorch import ToTensorV2
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_detector import decode_regression_to_boxes

from src.torch_implementation.dataset import PascalVOCDataset
import albumentations as A

os.environ["KERAS_BACKEND"] = "torch"

import json
import keras_cv

if __name__ == "__main__":

    class_mapping ={
        0: 'person'
    }

    xs_configuration_yolo_folder = r'C:\Users\tristan_cotte\Downloads\yolov8-keras-yolo_v8_xs_backbone_coco-v2.tar\yolov8-keras-yolo_v8_xs_backbone_coco-v2'
    config_file = os.path.join(xs_configuration_yolo_folder, 'config.json')

    with open(config_file, 'r') as JSON:
        json_dict = json.load(JSON)

    backbone = keras_cv.models.YOLOV8Backbone.from_config(json_dict['config'])
    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1,
    )

    print(model.summary())

    # yolo.to('cuda')

    transform = A.Compose([
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    train_dataset = PascalVOCDataset(
        data_folder=r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset_pascal_voc\Evry_dataset_2024_pascal_voc\VOCdevkit\VOC",
        split='train',
        transform=transform)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # For Training
    images, boxes, labels = next(iter(data_loader))
    # images = torch.Tensor(list(image for image in images))
    # # targets = [{k: v for k, v in t.items()} for t in boxes]
    # output = model(images, boxes, labels)
    #
    # # Returns total_loss and detections
    # print(output)
    #
    # decode_regression_to_boxes(output)

    y_pred = model(torch.permute(images, (0, 2, 3, 1)))
    box_pred, cls_pred = y_pred["boxes"], y_pred["classes"]
    pred_boxes = decode_regression_to_boxes(box_pred)
    print(pred_boxes)
    print(pred_boxes.size())
    print(cls_pred.size())

    # # For inference
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)  # Returns predictions
    # print(predictions[0])

