from functools import partial
from typing import Union, Tuple

import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def create_retinanet_model(
        score_threshold: float,
        iou_threshold: float,
        max_det: int = 300,
        num_classes: int = 91,
        use_COCO_pretrained_weights: bool = True,
        mean_values: Union[Tuple[float, float, float], None] = None,
        std_values: Union[Tuple[float, float, float], None] = None,
        unfrozen_layers: int = 3,
        trained_weights: Union[str, None] = None
):
    if trained_weights is None:
        if use_COCO_pretrained_weights:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        else:
            weights = None

    else:
        weights = None

    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=weights,
        image_mean=mean_values,
        image_std=std_values,
        score_thresh=score_threshold,
        nms_thresh=iou_threshold,
        detections_per_img=max_det,
        trainable_backbone_layers=unfrozen_layers
    )

    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )

    if trained_weights is not None:
        model.load_state_dict(torch.load(trained_weights, weights_only=True))

    return model


def create_faster_rcnn_model(num_classes=91):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))

# if __name__ == "__main__":
#     class_mapping = {
#         0: 'person'
#     }
#
#     xs_configuration_yolo_folder = r'C:\Users\tristan_cotte\Downloads\yolov8-keras-yolo_v8_xs_backbone_coco-v2.tar\yolov8-keras-yolo_v8_xs_backbone_coco-v2'
#     config_file = os.path.join(xs_configuration_yolo_folder, 'config.json')
#
#     with open(config_file, 'r') as JSON:
#         json_dict = json.load(JSON)
#
#     # yolo.to('cuda')
#
#     transform = A.Compose([
#         A.Normalize(),
#         A.RandomCrop(width=512, height=512),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         ToTensorV2()
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
#
#     train_dataset = PascalVOCDataset(
#         data_folder=r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset_pascal_voc\Evry_dataset_2024_pascal_voc\VOCdevkit\VOC",
#         split='train',
#         transform=transform)
#
#     data_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=2,
#         shuffle=True,
#         # collate_fn=train_dataset.collate_fn
#         collate_fn=collate_fn
#     )
#
#     model = create_retinanet_model(num_classes=len(class_mapping),
#                                    )
#
#     # For Training
#     images, target = next(iter(data_loader))
#     model(images, target)
