import json
import os
import time

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms

from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.model_retinanet import collate_fn, create_retinanet_model
from src.torch_implementation.utils import UnNormalize


def filter_predictions_by_confidence_threshold(predictions: dict, confidence_threshold: float):
    filter_by_confidence = predictions['scores'] > confidence_threshold
    for k, v in predictions.items():
        predictions[k] = predictions[k][filter_by_confidence]
    return predictions


def apply_nms_on_predictions(predictions: dict, iou_threshold: float):
    kept_bboxes = nms(boxes=predictions['boxes'], scores=predictions['scores'], iou_threshold=iou_threshold)

    for k, v in predictions.items():
        if predictions[k].size()[0] != 0:
            predictions[k] = torch.stack([predictions[k][i] for i in kept_bboxes.tolist()])
    # predictions['labels'] = predictions['labels'].type(torch.int64)
    return predictions


def apply_postprocess_on_predictions(predictions: list[dict], iou_threshold: float, confidence_threshold: float):
    post_processed_predictions = []
    for one_picture_prediction in predictions:
        one_picture_prediction = filter_predictions_by_confidence_threshold(predictions=one_picture_prediction,
                                                                            confidence_threshold=confidence_threshold)
        one_picture_prediction = apply_nms_on_predictions(predictions=one_picture_prediction,
                                                          iou_threshold=iou_threshold)
        post_processed_predictions.append(one_picture_prediction)
    return predictions


class_mapping = {
    0: 'person'
}

IMAGE_SIZE = (512, 512)

xs_configuration_yolo_folder = r'C:\Users\tristan_cotte\Downloads\yolov8-keras-yolo_v8_xs_backbone_coco-v2.tar\yolov8-keras-yolo_v8_xs_backbone_coco-v2'
config_file = os.path.join(xs_configuration_yolo_folder, 'config.json')

with open(config_file, 'r') as JSON:
    json_dict = json.load(JSON)

# yolo.to('cuda')

transform = A.Compose([
    A.Normalize(),
    A.RandomCrop(*IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

train_dataset = PascalVOCDataset(
    data_folder=r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset_pascal_voc\Evry_dataset_2024_pascal_voc\VOCdevkit\VOC",
    split='train',
    single_cls=True,
    transform=transform)

data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    # collate_fn=train_dataset.collate_fn
    collate_fn=collate_fn
)

# model = create_faster_rcnn_model(num_classes=2)
model = create_retinanet_model(num_classes=len(class_mapping))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

model.load_state_dict(torch.load('../../models/model_torch/retinanet.pth', weights_only=True))
model.to(device)
model.eval()

if __name__ == "__main__":
    MIN_CONFIDENCE = 0.1
    MIN_IOU_THRESHOLD = 0.3

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    metric = MeanAveragePrecision(iou_type="bbox")

    for i in range(4):
        images, targets = next(iter(data_loader))
        images = list(image.to(device) for image in images)

        targets_gpu = []
        for j in range(len(targets)):
            targets_gpu.append({k: v.to(device=device, non_blocking=True) for k, v in targets[0].items()})

        start_predictions = time.time()
        with torch.no_grad():
            predictions = model(images)
        print(f'Prediction time: {time.time() - start_predictions}')

        processed_predictions = apply_postprocess_on_predictions(
            predictions=predictions,
            iou_threshold=MIN_IOU_THRESHOLD,
            confidence_threshold=MIN_CONFIDENCE)

        metric.update(processed_predictions, targets_gpu)

        for index, (img, pred) in enumerate(zip(images, processed_predictions)):
            img = unorm(img)
            img = img.detach().cpu().numpy()
            img = img.transpose(1, 2, 0)

            for bbox_target in targets[index]['boxes']:
                bbox_target = bbox_target.detach().cpu().numpy()
                cv2.rectangle(img, (int(bbox_target[0]), int(bbox_target[1])),
                              (int(bbox_target[2]), int(bbox_target[3])), (255, 0, 0), 1)

            pred_scores = pred['scores'].detach().cpu().numpy()

            metric.update([pred], [targets_gpu[index]])

            for bbox_index in range(len(pred_scores)):
                bbox = pred['boxes'][bbox_index].detach().cpu().numpy()
                # bbox = ((int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])))

                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
                cv2.putText(img, f'{pred_scores[bbox_index]:.2f}', (int(bbox[0] - 10), int(bbox[1] - 10)),
                            color=(0, 0, 255), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)

            plt.imshow(img)
            plt.show()

    print(metric.compute())
