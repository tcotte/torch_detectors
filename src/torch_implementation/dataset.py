import imutils
import imutils.paths
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
import albumentations as A


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, single_cls, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard annotation_files that are considered difficult to detect?
        """
        self.split = split.upper()
        self._single_cls = single_cls

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        # Read data files
        # with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
        #     self.images = json.load(j)
        # with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
        #     self.annotation_files = json.load(j)
        self.images = list(imutils.paths.list_images(os.path.join(self.data_folder, 'JPEGImages')))

        annotation_folder = os.path.join(self.data_folder, 'Annotations')
        self.annotation_files = list([os.path.join(annotation_folder, i) for i in os.listdir(annotation_folder)])

        assert len(self.images) == len(self.annotation_files)

        self.class_mapping, self.number_obj_by_cls = self.get_class_mapping()

        self.transform = transform

    def get_class_mapping(self):
        list_classes = []
        dict_classes = {}

        for xml_file in tqdm(self.annotation_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.iter("object"):
                cls = obj.find("name").text
                if cls not in list_classes:
                    list_classes.append(cls)

                if cls not in dict_classes.keys():
                    dict_classes[cls] = 1
                else:
                    dict_classes[cls] += 1

        class_mapping = {}
        for idx, value in enumerate(list_classes):
            class_mapping[idx] = value

        if self._single_cls:
            class_mapping = {0: 'cls0'}
            dict_classes = {'cls0': sum(dict_classes.values())}

        return class_mapping, dict_classes

    def parse_annotation(self, xml_file, single_cls: bool = False):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_name = root.find("filename").text
        image_path = os.path.join(self.data_folder, 'JPEGImages', image_name)

        boxes = []
        classes = []
        for obj in root.iter("object"):
            cls = obj.find("name").text
            classes.append(cls)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            if xmin >= xmax:
                # print(f"Error loading bounding box in {image_name}")
                pass
            elif ymin >= ymax:
                pass
                # print(f"Error loading bounding box in {image_name}")
            else:
                boxes.append([xmin, ymin, xmax, ymax])

        if not single_cls:
            class_ids = [
                list(self.class_mapping.keys())[list(self.class_mapping.values()).index(cls)]
                for cls in classes
            ]
        else:
            class_ids = [0] * len(boxes)
        return {'boxes': boxes, 'labels': class_ids, 'image': image_path}

    def __getitem__(self, i):
        # Read annotation_files in this image (bounding boxes, labels, difficulties)
        objects = self.parse_annotation(xml_file=self.annotation_files[i], single_cls=True)

        # Read image
        image = Image.open(objects['image'], mode='r')
        image = image.convert('RGB')

        # difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Apply transformations
        transformed = self.transform(image=np.array(image),
                                     bboxes=np.array(objects['boxes']),
                                     class_labels=np.array(objects['labels']))
        image = transformed['image']
        boxes = torch.FloatTensor(transformed['bboxes'])  # (n_objects, 4)
        labels = torch.LongTensor(transformed['class_labels'])  # (n_objects)

        return image, {'boxes': boxes, 'labels': labels}

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of annotation_files, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels


if __name__ == "__main__":
    transform = A.Compose([
        A.RandomCrop(width=1500, height=1500),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    train_dataset = PascalVOCDataset(
        data_folder=r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset_pascal_voc\Evry_dataset_2024_pascal_voc\VOCdevkit\VOC",
        split='train',
        transform=transform)

    print(train_dataset[0])