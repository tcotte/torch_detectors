import os
import typing

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from picsellia import Client, Experiment
from picsellia.types.enums import InferenceType

from src.torch_implementation.dataset import PascalVOCTestDataset
import albumentations as A

from src.torch_implementation.model_retinanet import build_retinanet_model

load_dotenv('../.env')

if __name__ == '__main__':

    def fill_picsellia_evaluation_tab(model: torch.nn.Module, data_loader, experiment: Experiment,
                                      dataset_version_name: str,
                                      device, batch_size: int):
        dataset_version = experiment.get_dataset(name=dataset_version_name)
        picsellia_labels = dataset_version.list_labels()

        model.to(device)
        model.eval()

        for i, (images, file_paths) in enumerate(data_loader):
            images = list(image.to(device) for image in images)

            with torch.no_grad():
                predictions = model(images)

            for idx in range(batch_size):
                asset = dataset_version.find_asset(filename=file_paths[idx])

                resized_image_height, resized_image_width = images[idx].size()[-2:]
                original_height = asset.height
                original_width = asset.width

                # calculate the scale factors for width and height
                width_scale = original_width / resized_image_width
                height_scale = original_height / resized_image_height

                picsellia_rectangles: list = []
                for box, label, score in zip(predictions[idx]['boxes'], predictions[idx]['labels'],
                                             predictions[idx]['scores']):
                    box = box.cpu().numpy()
                    label = int(np.squeeze(label.cpu().numpy()))
                    score = float(np.squeeze(score.cpu().numpy()))
                    rectangle = (int(round(box[0] * width_scale)),
                                 int(round(box[1] * height_scale)),
                                 int(round((box[2] - box[0]) * width_scale)),
                                 int(round((box[3] - box[1]) * height_scale)),
                                 picsellia_labels[label],
                                 score)
                    picsellia_rectangles.append(rectangle)

                evaluation = experiment.add_evaluation(asset, rectangles=picsellia_rectangles)

                job = experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)
        job.wait_for_done()


    IMAGE_SIZE = (2048, 2048)
    BATCH_SIZE = 1
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    EXPERIMENT_ID = '019493b6-8e82-73ab-bb93-067dd78ecb18'
    NUM_WORKERS = 8

    picsellia_token = os.getenv('PICSELLIA_TOKEN')
    organisation_name = os.getenv('ORGANISATION_NAME')
    picsellia_client = Client(picsellia_token, organization_name=os.getenv('ORGANIZATION_NAME'))

    experiment = picsellia_client.get_experiment_by_id(EXPERIMENT_ID)
    experiment.delete_all_artifacts()
    experiment.delete_all_logs()

    print(experiment)
    dataset_version = experiment.get_dataset(name='test')

    path_dl_pix = '../picsellia_downloads/dataset_pics'

    if not os.path.exists(path_dl_pix):
        dataset_version.download(path_dl_pix)

    valid_transform = A.Compose([
        A.Resize(*IMAGE_SIZE),
        ToTensorV2()
    ])

    test_dataset = PascalVOCTestDataset(image_folder=path_dl_pix, transform=valid_transform)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_retinanet_model(
        score_threshold=0.2,
        iou_threshold=0.2,
        max_det=300,
        num_classes=2,
        mean_values=(0.9629258011853685, 1.1043921727662964, 0.9835339608076883),
        std_values=(0.08148765554920795, 0.10545005065566, 0.13757230267160245),
        trained_weights=r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\src\torch_implementation\models\latest_img_size_2048_classif.pth'
    )

    fill_picsellia_evaluation_tab(model=model, data_loader=test_data_loader, experiment=experiment, device=DEVICE,
                                  batch_size=BATCH_SIZE, dataset_version_name='test')
