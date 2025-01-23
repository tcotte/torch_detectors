import argparse
import os
import typing
from datetime import datetime

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_warmup import ExponentialWarmup
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.loggers.picsellia_logger import PicselliaLogger
from src.torch_implementation.model_retinanet import collate_fn, create_retinanet_model
from src.torch_implementation.utils import Averager, apply_postprocess_on_predictions, EarlyStopper, apply_loss_weights, \
    read_configuration_file


def train_model(model, optimizer, train_data_loader, val_data_loader, lr_scheduler, lr_warmup, nb_epochs,
                path_saved_models: str, loss_coefficients: dict, patience: int, callback):
    early_stopper = EarlyStopper(patience=patience)
    visualisation_val_loss = True
    best_map = 0.0
    metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)

    loss_training_hist = Averager()
    loss_validation_hist = Averager()

    for epoch in range(nb_epochs):
        loss_training_hist.reset()
        loss_validation_hist.reset()

        model.train()
        with tqdm(train_data_loader, unit="batch") as t_epoch:

            for images, targets in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")

                images = list(image.type(torch.FloatTensor).to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                '''
                Losses: 
                - Sigmoid focal loss for classification
                - l1 for regression
                '''
                loss_dict = model(images, targets)
                loss_dict = apply_loss_weights(loss_dict=loss_dict, loss_coefficients=loss_coefficients)

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

            if visualisation_val_loss:
                for images, targets in val_data_loader:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with torch.no_grad():
                        val_loss_dict = model(images, targets)

                    val_loss_dict = apply_loss_weights(loss_dict=val_loss_dict, loss_coefficients=loss_coefficients)
                    total_val_loss = sum(loss for loss in val_loss_dict.values())

                    loss_validation_hist.send({
                        "regression": val_loss_dict['bbox_regression'].item(),
                        "classification": val_loss_dict['classification'].item(),
                        "total": total_val_loss.item()
                    })

                if early_stopper.early_stop(validation_loss=total_val_loss.item()):
                    torch.save(model.state_dict(), os.path.join(path_saved_models, 'latest.pth'))
                    callback.on_train_end(best_validation_map=best_map, path_saved_models=path_saved_models)
                    break

            # update the learning rate
            with lr_warmup.dampening():
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

            # send targets to GPU
            targets_gpu = []
            for j in range(len(targets)):
                targets_gpu.append({k: v.to(device=device, non_blocking=True) for k, v in targets[j].items()})

            metric.update(processed_predictions, targets_gpu)

        validation_metrics = metric.compute()

        # TODO display precision / recall in Picsellia interface

        print(f"Epoch #{epoch} Training loss: {loss_training_hist.value} "
              f"Validation loss {loss_validation_hist.value}"
              f"- Accuracies: 'mAP' {float(validation_metrics['map']):.3} / "
              f"'mAP[50]': {float(validation_metrics['map_50']):.3} / "
              f"'mAP[75]': {float(validation_metrics['map_75']):.3} /"
              f"'Precision': {float(validation_metrics['precision']):.3} / "
              f"'Recall': {float(validation_metrics['recall']):.3} ")
        if validation_metrics['map'] >= best_map:
            best_map = float(validation_metrics['map'])
            torch.save(model.state_dict(), os.path.join(path_saved_models, 'best.pth'))

        callback.on_epoch_end(training_losses=loss_training_hist.value,
                              validation_losses=loss_validation_hist.value,
                              accuracies={
                                  'map': float(validation_metrics['map']),
                                  'mAP[50]': float(validation_metrics['map_50']),
                                  'mAP[75]': float(validation_metrics['map_75']),
                                  'precision': float(validation_metrics['precision']),
                                  'recall': float(validation_metrics['recall'])
                              },
                              current_lr=optimizer.param_groups[0]['lr'])
        metric.reset()

    torch.save(model.state_dict(), os.path.join(path_saved_models, 'latest.pth'))
    callback.on_train_end(best_validation_map=best_map, path_saved_models=path_saved_models)


# todo:  add possibility to add path to models
parser = argparse.ArgumentParser(
    prog='RetinaNet_Trainer',
    description='The aim of this program is to train RetinaNet model with custom dataset',
    epilog='------- SGS France - Operational Innovation -------')

parser.add_argument('-epoch', '--epoch', type=int, default=100, required=False,
                    help='Number of epochs used for train the model')
parser.add_argument('-name', '--run_name', type=str, default='', required=False,
                    help='Experiment name')
parser.add_argument('-configfile', '--configuration_file', type=str, default='', required=False,
                    help='Advanced configurations file path')
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
parser.add_argument( '--unfreeze', type=int, default=2, required=False,
                    help='Number of trainable layers starting from final block')
parser.add_argument( '--patience', type=int, default=100, required=False,
                    help='Number of epochs to wait without improvement in validation loss before early stopping the '
                         'training')

args = parser.parse_args()

assert(0 <= args.unfreeze <= 5, f"Number of unfrozen backbone layers has to lie between 0 and 5, got: {args.unfreeze}")

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
        # A.Normalize(mean=[0.9629258011853685, 1.1043921727662964, 0.9835339608076883],
        #             std=[0.08148765554920795, 0.10545005065566, 0.13757230267160245],
        #             max_pixel_value=207),
        A.RandomCrop(*IMAGE_SIZE),
        A.HueSaturationValue(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    # train_transform = A.Compose([
    #     A.Normalize(mean=[0.9629258011853685, 1.1043921727662964, 0.9835339608076883],
    #                 std=[0.08148765554920795, 0.10545005065566, 0.13757230267160245],
    #                 max_pixel_value=207),
    #     A.RandomCrop(*IMAGE_SIZE),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.OneOf([
    #         A.GaussNoise(std_range=(0.2, 0.44), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1, p=0.5),
    #         A.GridDropout(
    #             ratio=0.1,
    #             unit_size_range=None,
    #             random_offset=True,
    #             p=0.2),
    #     ]),
    #     A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.2, 0.2), p=0.5),
    #     A.OneOf([
    #         A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    #         A.HueSaturationValue(p=0.1),
    #         A.RandomBrightnessContrast(p=0.2),
    #     ]),
    #     A.OneOf([
    #         A.ElasticTransform(alpha=1, sigma=50, interpolation=1, approximate=False, same_dxdy=False,
    #                            mask_interpolation=0, noise_distribution='gaussian', p=0.2),
    #         A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), interpolation=1, normalized=True,
    #                          mask_interpolation=0, p=0.3)
    #     ]),
    #     A.RandomScale(scale_limit=(-0.3, 0.3), interpolation=1, mask_interpolation=0, p=0.5),
    #
    #     ToTensorV2()
    # ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    valid_transform = A.Compose([
        # A.Normalize(mean=[0.9629258011853685, 1.1043921727662964, 0.9835339608076883],
        #             std=[0.08148765554920795, 0.10545005065566, 0.13757230267160245],
        #             max_pixel_value=207),
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

    if args.configuration_file is not None:
        configs = read_configuration_file(args.configuration_file)
    else:
        configs = {
            'loss': {
                'bbox_regression': 1,
                'classification': 1
            },
            'augmentations': {
                    'normalization': {
                        'mean': None,
                        'std': None
                    }
                },
            'learning_rate': {
                'warmup': {
                    'last_step': 1,
                    'warmup_period': 1
                }

            }
        }

    # TODO concatenate class mappings of train and test datasets
    class_mapping = train_dataset.class_mapping

    model = create_retinanet_model(num_classes=len(class_mapping),
                                   use_pretrained_weights=args.pretrained_weights,
                                   score_threshold=args.confidence_threshold,
                                   iou_threshold=args.iou_threshold,
                                   unfrozen_layers=args.unfreeze,
                                   mean_values=configs['augmentations']['normalization']['mean'],
                                   std_values=configs['augmentations']['normalization']['std']
                                   )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    warmup_scheduler = ExponentialWarmup(optimizer,
                                         warmup_period=configs['learning_rate']['warmup']['warmup_period'],
                                         last_step=configs['learning_rate']['warmup']['last_step']
                                         )

    run_name = args.run_name if args.run_name != '' else None

    picsellia_logger = PicselliaLogger(path_env_file=PATH_ENV_FILE, run_name=run_name)
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
        'lr_scheduler_gamma': args.lr_gamma,
        'loss_weight_regression': configs['loss']['bbox_regression'],
        'loss_weight_classification': configs['loss']['classification'],
        'normalization_mean': model.transform.image_mean,
        'normalization_std': model.transform.image_std,
        'unfrozen_backbone_layers': args.unfreeze,
        'used_GPU': torch.cuda.get_device_name(),
        'patience': args.patience,
        'exp_warmup_last_step': configs['learning_rate']['warmup']['last_step'],
        'exp_warmup_period': configs['learning_rate']['warmup']['warmup_period']
    }

    picsellia_logger.on_train_begin(params=params, class_mapping=class_mapping)

    train_model(model, optimizer, train_data_loader, val_data_loader, lr_scheduler, warmup_scheduler, NB_EPOCHS, PATH_SAVED_MODELS,
                loss_coefficients=configs['loss'], callback=picsellia_logger, patience=args.patience)
