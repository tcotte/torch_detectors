import time
import typing

import albumentations as A
import cv2
import imutils.paths
import matplotlib.pyplot as plt
import numpy as np
import torch
from albucore import normalize_opencv, normalize
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection import MeanAveragePrecision

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
from PIL import Image

from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.model_retinanet import collate_fn, create_retinanet_model
from src.torch_implementation.utils import UnNormalize, apply_postprocess_on_predictions

class_mapping = {
    0: 'person'
}

DATASET_PATH: typing.Final[str] = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\test"
MODEL_PATH: typing.Final[
    str] = r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\src\torch_implementation\models\best_normalization_custom_retinanet.pth'
IMAGE_SIZE: typing.Final[tuple[int, int]] = (1024, 1024)
MIN_CONFIDENCE = 0.4
MIN_IOU_THRESHOLD = 0.01

#
# transform = A.Compose([
#     # A.Normalize(),
#     A.RandomCrop(*IMAGE_SIZE),
#     # A.HorizontalFlip(p=0.5),
#     # A.RandomBrightnessContrast(p=0.2),
#     ToTensorV2()
# ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))
#
# train_dataset = PascalVOCDataset(
#     data_folder=DATASET_PATH,
#     split='train',
#     single_cls=True,
#     transform=transform)
#
# data_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=4,
#     shuffle=True,
#     # collate_fn=train_dataset.collate_fn
#     collate_fn=collate_fn
# )


# model = create_faster_rcnn_model(num_classes=2)
model = create_retinanet_model(num_classes=len(class_mapping),
                               use_pretrained_weights=True,
                               score_threshold=MIN_IOU_THRESHOLD,
                               iou_threshold=MIN_CONFIDENCE,
                               unfrozen_layers=3,
                               mean_values=(0.9629258011853685, 1.1043921727662964, 0.9835339608076883),
                               std_values=(0.08148765554920795, 0.10545005065566, 0.13757230267160245)
                               )

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.to(device)
model.eval()

if __name__ == "__main__":


    # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    metric = MeanAveragePrecision(iou_type="bbox")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='torchvision',
        model=model,
        confidence_threshold=MIN_CONFIDENCE,
        image_size=1024,
        device="cpu" if not torch.cuda.is_available() else "cuda:0",
        load_at_init=True,
    )

    for i in range(5):
        img0 = list(imutils.paths.list_images(DATASET_PATH))[12 + i]

        image = Image.open(img0, mode='r')
        image = image.convert('RGB')
        # norm_image = normalize(img=np.array(image),
        #                        mean=np.array([0.485, 0.456, 0.406], dtype=np.float32)*255.0,
        #                        denominator=np.reciprocal(np.array([0.229, 0.224, 0.225], dtype=np.float32)*255.0))

        start_prediction = time.time()

        result = get_sliced_prediction(
            np.array(image).astype(np.uint8),
            detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type='NMS'
        )

        print(f"Time taken to predict {time.time() - start_prediction:.2f} seconds")

        result.export_visuals(export_dir='../../Output_SAHI',
                              rect_th=1,
                              hide_labels=True,
                              hide_conf=False,
                              file_name=f'{str(i)}')
