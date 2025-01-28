import os
import shutil
import zipfile
from typing import Union

import torch
import yaml
import albumentations as A
from albumentations import ToTensorV2
from dotenv import load_dotenv
import picsellia
from picsellia import Client, ModelVersion, DatasetVersion
from picsellia.types.enums import AnnotationFileType
from pytorch_warmup import ExponentialWarmup
from torch.utils.data import DataLoader

from picsellia_integration.retinanet_parameters import TrainingParameters
from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.loggers.picsellia_logger import PicselliaLogger
from src.torch_implementation.model_retinanet import collate_fn, build_retinanet_model
from collections.abc import MutableMapping


def extract_zipfile(path_input_zip: str, directory_to_extract_to: str) -> None:
    def refactor_member_zip_name(member_name: str) -> str:
        return "_".join(member_name.split('_')[1:]).replace('_annotations', '')

    zip_ref = zipfile.ZipFile(path_input_zip)
    list_zip_info = zip_ref.infolist()

    for zip_info in list_zip_info:
        zip_info.filename = refactor_member_zip_name(member_name=zip_info.filename)
        zip_ref.extract(zip_info, path=directory_to_extract_to)


def download_annotations(dataset_version: picsellia.DatasetVersion, annotation_folder_path: str):
    zip_file = dataset_version.export_annotation_file(AnnotationFileType.PASCAL_VOC, "./")
    extract_zipfile(path_input_zip=zip_file, directory_to_extract_to=annotation_folder_path)
    shutil.rmtree(os.path.dirname(os.path.dirname(zip_file)))


def download_datasets(dataset_versions: list[picsellia.DatasetVersion], root_folder: str = 'dataset'):
    '''
        .
    ├── train/
    │   ├── JPEGImages/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── Annotations/
    │   │   ├── img1.xml
    │   │   ├── img2.xml
    │   │   └── ...
    │   └── ...
    └── test/
        ├── JPEGImages/
        │   └── ...jpg
        └── Annotations/
            └── ...xml
    '''
    root = root_folder
    for dataset in dataset_versions:
        annotations_folder_path = os.path.join(root, dataset.version, 'Annotations')
        images_folder_path = os.path.join(root, dataset.version, 'JPEGImages')

        os.makedirs(images_folder_path)
        os.makedirs(annotations_folder_path)

        assets = dataset.list_assets()
        assets.download(images_folder_path, max_workers=os.cpu_count())

        download_annotations(dataset_version=dataset, annotation_folder_path=annotations_folder_path)


def download_experiment_file(base_model: ModelVersion, experiment_file: str):
    comport_file: bool = any([i.name == experiment_file for i in base_model.list_files()])
    if comport_file:
        base_model_weights = base_model.get_file(experiment_file)
        base_model_weights.download()


def get_advanced_config(base_model: ModelVersion) -> Union[None, dict]:
    download_experiment_file(base_model, 'config')
    config_filename = base_model.get_file('config').filename
    with open(config_filename) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def create_dataloaders(image_size: tuple[int, int], single_cls: bool, num_workers: int, batch_size: int) -> \
        tuple[DataLoader, DataLoader, PascalVOCDataset, PascalVOCDataset]:
    train_transform = A.Compose([
        A.RandomCrop(*image_size),
        A.HueSaturationValue(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    valid_transform = A.Compose([
        A.RandomCrop(*image_size),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

    train_dataset = PascalVOCDataset(
        data_folder=os.path.join('../', 'dataset', 'train'),
        split='train',
        single_cls=single_cls,
        transform=train_transform)

    val_dataset = PascalVOCDataset(
        data_folder=os.path.join('../', 'dataset', 'train'),
        split='test',
        single_cls=single_cls,
        transform=valid_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_data_loader, val_data_loader, train_dataset, val_dataset


def get_class_mapping_from_picsellia(dataset_versions: list[DatasetVersion], single_cls: bool = False) -> dict[
    int, str]:
    labels = ['bckg']

    if not single_cls:
        for ds_version in dataset_versions:
            for label in ds_version.list_labels():
                if label not in labels:
                    labels.append(label.name)

    else:
        labels.append('cls0')

    return dict(zip(range(len(labels)), labels))


def get_optimizer(optimizer_name: str, lr0: float, weight_decay: float, model_params) -> torch.optim.Optimizer:
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr=lr0, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model_params, lr=lr0, weight_decay=weight_decay)

    return optimizer


def flatten_dictionary(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dictionary(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


if __name__ == "__main__":
    load_dotenv('../.env')

    api_token = os.getenv('PICSELLIA_TOKEN')
    organization_name = os.getenv('ORGANIZATION_NAME')

    client = Client(api_token, organization_name=organization_name)

    experiment = client.get_experiment_by_id(id='0194a70c-c596-7e1a-af33-7e6f8bbeb4b5')

    datasets = experiment.list_attached_dataset_versions()

    # Download datasets
    # download_datasets(dataset_versions=datasets)

    base_model = experiment.get_base_model_version()

    # Download weights
    download_experiment_file(base_model=base_model, experiment_file='weights')

    # Download parameters
    parameters = experiment.get_log(name='parameters').data
    single_cls = bool(parameters['single_cls'])

    IMG_SIZE = parameters['image_size']
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']

    # Download advanced configurations from *config* file
    advanced_config = get_advanced_config(base_model=base_model)
    print(advanced_config)

    total_configs = parameters | advanced_config

    # TODO change name of learning_rate parameter
    if 'learning_rate' in parameters.keys():
        total_configs['learning_rate']['initial_lr'] = parameters['learning_rate']

    training_parameters = TrainingParameters(**total_configs)

    # Create dataloaders
    train_dataloader, val_dataloader, train_dataset, val_dataset = create_dataloaders(
        image_size=training_parameters.image_size,
        single_cls=training_parameters.single_class,
        num_workers=training_parameters.workers_number,
        batch_size=training_parameters.batch_size
    )

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    class_mapping = get_class_mapping_from_picsellia(dataset_versions=datasets,
                                                     single_cls=training_parameters.single_class)

    # Build model
    model = build_retinanet_model(num_classes=len(class_mapping),
                                  use_COCO_pretrained_weights=training_parameters.coco_pretrained_weights,
                                  score_threshold=training_parameters.confidence_threshold,
                                  iou_threshold=training_parameters.iou_threshold,
                                  unfrozen_layers=training_parameters.unfreeze,
                                  mean_values=training_parameters.augmentations.normalization.mean,
                                  std_values=training_parameters.augmentations.normalization.std
                                  )

    optimizer = get_optimizer(optimizer_name=training_parameters.optimizer,
                              lr0=training_parameters.learning_rate.initial_lr,
                              weight_decay=training_parameters.weight_decay,
                              model_params=model.parameters())

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=training_parameters.learning_rate.decay.step_size,
                                                   gamma=training_parameters.learning_rate.decay.gamma)
    warmup_scheduler = ExponentialWarmup(optimizer,
                                         warmup_period=training_parameters.learning_rate.warmup.warmup_period,
                                         last_step=training_parameters.learning_rate.warmup.last_step
                                         )

    # Logger
    picsellia_logger = PicselliaLogger.from_picsellia_client_and_experiment(picsellia_experiment=experiment,
                                                                            picsellia_client=client)
    picsellia_logger.log_split_table(
        annotations_in_split={"train": len(train_dataset), "val": len(val_dataset)},
        title="Nb elem / split")

    picsellia_logger.log_split_table(annotations_in_split=train_dataset.number_obj_by_cls, title="Train split")
    picsellia_logger.log_split_table(annotations_in_split=val_dataset.number_obj_by_cls, title="Val split")
    picsellia_logger.on_train_begin(params=flatten_dictionary(dictionary=training_parameters.dict()),
                                    class_mapping=class_mapping)



