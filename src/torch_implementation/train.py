import os
import typing
from datetime import datetime

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.detect import apply_postprocess_on_predictions
from src.torch_implementation.loggers.picsellia_logger import PicselliaLogger
from src.torch_implementation.model_retinanet import collate_fn, create_retinanet_model
from src.torch_implementation.utils import Averager


def train_model(model, optimizer, train_data_loader, val_data_loader, lr_scheduler, nb_epochs, path_saved_models: str,
                callback):
    best_map = 0.0
    metric = MeanAveragePrecision(iou_type="bbox")

    for epoch in range(nb_epochs):
        loss_training_hist = Averager()
        loss_training_hist.reset()

        model.train()
        with tqdm(train_data_loader, unit="batch") as t_epoch:

            for images, targets in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                '''
                Losses: 
                - Sigmoid focal loss for classification
                - l1 for regression
                '''
                loss_dict = model(images, targets)

                total_loss = sum(loss for loss in loss_dict.values())
                total_loss_value = total_loss.item()

                loss_training_hist.send({
                    "regression": loss_dict['bbox_regression'].item(),
                    "classification": loss_dict['classification'].item(),
                    "total": total_loss_value
                })  # Average out the loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                t_epoch.set_postfix(
                    total_loss=loss_training_hist.value['total'],
                    bbox_loss=loss_training_hist.value['regression'],
                    cls_loss=loss_training_hist.value['classification']
                )

            # update the learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

        # get_CUDA_memory_allocation()

        model.eval()

        for images, targets in val_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                predictions = model(images)

            processed_predictions = apply_postprocess_on_predictions(
                predictions=predictions,
                iou_threshold=MIN_IOU_THRESHOLD,
                confidence_threshold=MIN_CONFIDENCE)

            targets_gpu = []
            for j in range(len(targets)):
                targets_gpu.append({k: v.to(device=device, non_blocking=True) for k, v in targets[0].items()})

            metric.update(processed_predictions, targets_gpu)

        validation_metrics = metric.compute()
        print(f"Epoch #{epoch} loss: {loss_training_hist.value} "
              f"- Accuracies: 'mAP' {float(validation_metrics['map']):.3} / "
              f"'mAP[50]': {float(validation_metrics['map_50']):.3} / "
              f"'mAP[75]': {float(validation_metrics['map_75']):.3}")
        if validation_metrics['map'] > best_map:
            best_map = validation_metrics['map']
            torch.save(model.state_dict(), os.path.join(path_saved_models, 'best.pth'))

        callback.on_epoch_end(losses=loss_training_hist.value,
                              accuracies={
                                  'map': float(validation_metrics['map']),
                                  'mAP[50]': float(validation_metrics['map_50']),
                                  'mAP[75]': float(validation_metrics['map_75'])
                              })
        metric.reset()

    torch.save(model.state_dict(), os.path.join(path_saved_models, 'latest.pth'))
    callback.on_train_end(best_validation_map=best_map, path_saved_models=path_saved_models)


MIN_CONFIDENCE: typing.Final[float] = 0.2
MIN_IOU_THRESHOLD: typing.Final[float] = 0.2
BATCH_SIZE: typing.Final[int] = 1

IMAGE_SIZE: typing.Final[tuple[int, int]] = (128, 128)

DATA_TRAIN_DIR: typing.Final[str] = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\train"
DATA_VALIDATION_DIR: typing.Final[str] = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\test"

LEARNING_RATE: typing.Final[float] = 0.001
NB_EPOCHS: typing.Final[int] = 20
NB_CLASSES: typing.Final[int] = 1

SINGLE_CLS: typing.Final[bool] = True

PATH_ENV_FILE: typing.Final[str] = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\.env"

if __name__ == "__main__":
    date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    PATH_SAVED_MODELS: typing.Final[str] = f"models/run_{date_time}"
    os.makedirs(PATH_SAVED_MODELS, exist_ok=True)

    train_transform = A.Compose([
        A.Normalize(),
        A.RandomCrop(*IMAGE_SIZE),
        A.HueSaturationValue(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    valid_transform = A.Compose([
        A.Normalize(),
        A.RandomCrop(*IMAGE_SIZE),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    train_dataset = PascalVOCDataset(
        data_folder=DATA_TRAIN_DIR,
        split='train',
        single_cls=SINGLE_CLS,
        transform=train_transform)

    val_dataset = PascalVOCDataset(
        data_folder=DATA_VALIDATION_DIR,
        split='test',
        single_cls=SINGLE_CLS,
        transform=valid_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = create_retinanet_model(num_classes=NB_CLASSES)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    picsellia_logger = PicselliaLogger(path_env_file=PATH_ENV_FILE)
    picsellia_logger.log_split_table(
        annotations_in_split={"train": len(train_data_loader), "val": len(val_data_loader)},
        title="Nb elem / split")
    picsellia_logger.log_split_table(annotations_in_split=train_dataset.number_obj_by_cls, title="Train split")
    picsellia_logger.log_split_table(annotations_in_split=val_dataset.number_obj_by_cls, title="Val split")

    params = {
        'minimum_confidence': MIN_CONFIDENCE,
        'minimum_iou': MIN_IOU_THRESHOLD,
        'batch_size': BATCH_SIZE,
        'image_size': IMAGE_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NB_EPOCHS,
        'nb_classes': NB_CLASSES,
        'single_class': SINGLE_CLS
    }

    # TODO concatenate class mappings of train and test datasets
    picsellia_logger.on_train_begin(params=params, class_mapping=train_dataset.class_mapping)

    train_model(model, optimizer, train_data_loader, val_data_loader, lr_scheduler, NB_EPOCHS, PATH_SAVED_MODELS,
                callback=picsellia_logger)
