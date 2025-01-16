import argparse
import os
import typing
from datetime import datetime

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.loggers.picsellia_logger import PicselliaLogger
from src.torch_implementation.model_retinanet import collate_fn, create_retinanet_model
from src.torch_implementation.utils import Averager, apply_postprocess_on_predictions


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
        if validation_metrics['map'] >= best_map:
            best_map = float(validation_metrics['map'])
            torch.save(model.state_dict(), os.path.join(path_saved_models, 'best.pth'))

        callback.on_epoch_end(losses=loss_training_hist.value,
                              accuracies={
                                  'map': float(validation_metrics['map']),
                                  'mAP[50]': float(validation_metrics['map_50']),
                                  'mAP[75]': float(validation_metrics['map_75'])
                              },
                              current_lr=optimizer.param_groups[0]['lr'])
        metric.reset()

    torch.save(model.state_dict(), os.path.join(path_saved_models, 'latest.pth'))
    callback.on_train_end(best_validation_map=best_map, path_saved_models=path_saved_models)

# todo:  add possibility to add path to models - add several workers to load data
parser = argparse.ArgumentParser(
    prog='RetinaNet_Trainer',
    description='The aim of this program is to train RetinaNet model with custom dataset',
    epilog='------- SGS France - Operational Innovation -------')

parser.add_argument('-epoch', '--epoch', type=int, default=100, required=False,
                    help='Number of epochs used for train the model')
parser.add_argument('-bs', '--batch_size', type=int, default=2, required=False,
                    help='Batch size during the training')
parser.add_argument('-device', '--device', type=str, default="cuda", required=False,
                    help='Device used to train the model')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, required=False,
                    help='Initial learning rate used for training')
parser.add_argument('--lr_step_size', type=int, default=50, required=False,
                    help='Period of learning rate decay')
parser.add_argument('--lr_gamma', type=float, default=0.1, required=False,
                    help='Multiplicative factor of learning rate decay')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005, required=False,
                    help='Weight decay used for training')
parser.add_argument('-env', '--path_env_file', type=str, required=True,
                    help='Path of .env file necessary to get Picsellia credentials')
parser.add_argument('-opt', '--optimizer', type=str, default='Adam',
                    help='Optimizer used for training')
parser.add_argument('-tds', '--train_dataset', type=str, required=True,
                    help='Path of training PASCAL_VOC dataset folder')
parser.add_argument('-vds', '--valid_dataset', type=str, required=True,
                    help='Path of validation PASCAL_VOC dataset folder')
parser.add_argument('-numw', '--num_workers', type=int, default=8, required=False,
                    help='Number of workers to retrieve data during training')
parser.add_argument('-imgsz', '--image_size', nargs='+', type=int,
                    help='Training image size')
parser.add_argument("-sglcls", "--single_class", default=False, action="store_true", required=False,
                    help="Use only one class")
parser.add_argument("--pretrained_weights", default=False, action="store_true", required=False,
                    help="Load model with pretrained weights from COCO dataset")
parser.add_argument('-conf', '--confidence_threshold', type=float, default=0.2, required=False,
                    help='Confidence threshold used to evaluate model')
parser.add_argument('-iou', '--iou_threshold', type=float, default=0.2, required=False,
                    help='IoU threshold used to evaluate model when NMS is applied')

args = parser.parse_args()

MIN_CONFIDENCE: typing.Final[float] = args.confidence_threshold
MIN_IOU_THRESHOLD: typing.Final[float] = args.iou_threshold
BATCH_SIZE: typing.Final[int] = args.batch_size

# noinspection PyTypeChecker
IMAGE_SIZE: typing.Final[tuple[int, int]] = tuple(args.image_size)

DATA_TRAIN_DIR: typing.Final[str] = args.train_dataset
DATA_VALIDATION_DIR: typing.Final[str] = args.valid_dataset

LEARNING_RATE: typing.Final[float] = args.learning_rate
NB_EPOCHS: typing.Final[int] = args.epoch

SINGLE_CLS: typing.Final[bool] = args.single_class

PATH_ENV_FILE: typing.Final[str] = args.path_env_file

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
        num_workers=args.num_workers,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # TODO concatenate class mappings of train and test datasets
    class_mapping = train_dataset.class_mapping

    model = create_retinanet_model(num_classes=len(class_mapping), use_pretrained_weights=args.pretrained_weights)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    picsellia_logger = PicselliaLogger(path_env_file=PATH_ENV_FILE)
    picsellia_logger.log_split_table(
        annotations_in_split={"train": len(train_dataset), "val": len(val_dataset)},
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
        'nb_classes': len(class_mapping),
        'single_class': SINGLE_CLS,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'pretrained_weights': args.pretrained_weights,
        'num_workers': args.num_workers,
        'lr_scheduler_step_size': args.lr_step_size,
        'lr_scheduler_gamma': args.lr_gamma
    }

    picsellia_logger.on_train_begin(params=params, class_mapping=class_mapping)

    train_model(model, optimizer, train_data_loader, val_data_loader, lr_scheduler, NB_EPOCHS, PATH_SAVED_MODELS,
                callback=picsellia_logger)
