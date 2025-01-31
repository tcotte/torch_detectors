import os
import random

import numpy as np
import torch
from albumentations import ToTensorV2
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import _default_anchorgen, RetinaNet, retinanet_resnet50_fpn_v2
import albumentations as A

from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.model_retinanet import collate_fn


def define_box_shape(box):
    return int(box[2] - box[0]), int(box[3] - box[1])


IMAGE_SIZE = (2048, 2048)
SINGLE_CLS = True
DATA_VALIDATION_DIR = r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\test'
BATCH_SIZE = 1
NB_SHOWN_BBOXES: int = 100


valid_transform = A.Compose([
    # A.RandomCrop(*IMAGE_SIZE),
    A.Resize(*IMAGE_SIZE),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

val_dataset = PascalVOCDataset(
    data_folder=DATA_VALIDATION_DIR,
    split='test',
    single_cls=SINGLE_CLS,
    add_bckd_as_class=False,
    transform=valid_transform)

val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    num_workers=os.cpu_count(),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


if __name__ == "__main__":
    use_image_from_dataset: bool = True
    index_image: int = 1
    anchor_size_coefficient: float = 1



    # anchor_generator: AnchorGenerator = _default_anchorgen()
    # print(anchor_generator.sizes)
    # print(anchor_generator.aspect_ratios)

    anchor_generator = AnchorGenerator((np.array(_default_anchorgen().sizes) * anchor_size_coefficient).tolist(),
                                       _default_anchorgen().aspect_ratios)
    # print(anchor_generator.cell_anchors)

    if not use_image_from_dataset:
        image = Tensor(3, *IMAGE_SIZE)

    else:
        image = val_dataset[index_image][0]
        # Put image into the correct format for the model
        input_image = image.unsqueeze(0)

    # Load model
    model = retinanet_resnet50_fpn_v2(pretrained=False)
    model.anchor_generator = anchor_generator
    model.eval()

    image_list = ImageList(tensors=input_image, image_sizes= list(IMAGE_SIZE))

    # Forward pass
    with torch.no_grad():
        features = model.backbone(input_image)
        features = list(features.values())
        anchors = model.anchor_generator(image_list = image_list, feature_maps = features)[0]  # Generate anchors based on feature maps

    # Visualize the anchor boxes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image.permute(1, 2, 0))  # If image is a tensor (C, H, W), permute it to (H, W, C)

    number_of_anchors = len(anchors)
    rdm_indexes = random.sample(list(range(number_of_anchors)), NB_SHOWN_BBOXES)

    colors = ['r', 'b']

    # Iterate over anchor boxes and plot them
    boxes = [anchors[i] for i in rdm_indexes]
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
