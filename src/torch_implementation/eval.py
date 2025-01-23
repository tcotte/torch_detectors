import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.model_retinanet import collate_fn, create_retinanet_model
from src.torch_implementation.utils import apply_postprocess_on_predictions, get_GPU_occupancy


def plot_precision_recall_curve(validation_metrics: dict, recall_thresholds: list[float]) -> plt.plot:
    recall_values = np.array(recall_thresholds)
    precision_values = np.array(
        [validation_metrics['precision'][0][i][0][0][-1] for i in range(len(recall_thresholds))])

    f1_scores = 2 * (precision_values * recall_values) / (precision_values + recall_values)
    best_threshold_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_threshold_index]

    fig, ax = plt.subplots()

    ax.plot(precision_values, recall_values, label=f'best F1: {best_f1:.2f}')

    ax.axhline(y=precision_values[best_threshold_index], color="red", linestyle='--', lw=1)
    ax.axvline(x=recall_values[best_threshold_index], color="red", linestyle='--', lw=1)

    yticks = [*ax.get_yticks(), precision_values[best_threshold_index]]
    yticklabels = [*ax.get_yticklabels(), float(round(precision_values[best_threshold_index], 2))]
    ax.set_yticks(yticks, labels=yticklabels)

    xticks = [*ax.get_xticks(), recall_values[best_threshold_index]]
    xticklabels = [*ax.get_xticklabels(), float(round(recall_values[best_threshold_index], 2))]
    ax.set_xticks(xticks, labels=xticklabels)

    ax.set_title('Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend(loc='lower left')

    return ax


def evaluate(model, eval_dataloader, device):
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)

    for images, targets in tqdm(eval_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)

        # send targets to GPU
        targets_gpu = []
        for j in range(len(targets)):
            targets_gpu.append({k: v.to(device=device, non_blocking=True) for k, v in targets[j].items()})

        metric.update(predictions, targets_gpu)

        print(get_GPU_occupancy())

    validation_metrics = metric.compute()

    return validation_metrics


if __name__ == '__main__':
    IMAGE_SIZE = (1024, 1024)
    SINGLE_CLS = True
    DATA_VALIDATION_DIR = r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\test'
    BATCH_SIZE = 8
    MIN_IOU_THRESHOLD = 0.2
    MIN_CONFIDENCE = 0.2





    valid_transform = A.Compose([
        A.RandomCrop(*IMAGE_SIZE),
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

    model = create_retinanet_model(num_classes=1,
                                   use_COCO_pretrained_weights=False,
                                   score_threshold=0.2,
                                   iou_threshold=0.2,
                                   trained_weights=r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\src\torch_implementation\models\best_normalization_custom_retinanet.pth',
                                   mean_values=(0.9629258011853685, 1.1043921727662964, 0.9835339608076883),
                                   std_values=(0.08148765554920795, 0.10545005065566, 0.13757230267160245)
                                   )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    model.eval()

    eval_metrics = evaluate(model, val_data_loader, device=device)

    f = plot_precision_recall_curve(validation_metrics=eval_metrics,
                                recall_thresholds=torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist())
    plt.savefig('precision_recall_curve.png')