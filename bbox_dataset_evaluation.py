import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.ops import box_iou


def get_boxes_from_annotation_file(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

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
    return boxes


def KMeans_clustering_anchor_boxes(bboxes: torch.tensor, k: int, stop_iter: int = 5) -> torch.tensor:
    # bboxes: x1, y1, x2, y2
    pbar = tqdm(total=stop_iter)

    rows = bboxes.shape[0]
    distances = torch.empty((rows, k))
    last_clusters = torch.zeros((rows,))

    cluster_indxs = np.random.choice(rows, k, replace=False)  # choose unique indexs in rows
    anchors = bboxes[cluster_indxs].clone()

    iteration = 0
    while True:
        # calculate the distances
        distances = IoU(anchors=anchors, bboxes=bboxes)

        nearest_clusters = torch.argmax(distances, dim=1)  # 0, 1, 2 ... K

        if (last_clusters == nearest_clusters).all():  # break if nothing changes
            iteration += 1
            if iteration == stop_iter:
                break
        else:
            iteration = 0

        # Take the mean and step for cluster coordiantes
        for cluster in range(k):
            anchors[cluster] = torch.mean(
                bboxes[nearest_clusters == cluster],
                axis=0
            )

        last_clusters = nearest_clusters.clone()
        pbar.update(1)
    pbar.close()
    return anchors


def IoU(anchors: torch.tensor, bboxes: torch.tensor) -> float:
    iou_values = box_iou(bboxes, anchors)
    return iou_values


if __name__ == "__main__":
    # DATASET_PATH: str = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\test"
    LABELS_PATH: str = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\test\Annotations"

    boxes = []

    for i in tqdm(os.listdir(LABELS_PATH)):
        if i.endswith(".xml"):
            annotation_file = os.path.join(LABELS_PATH, i)

            boxes.extend(get_boxes_from_annotation_file(xml_file=annotation_file))

    print("ok")

    boxes = np.array(boxes)
    width_boxes = boxes[:, 2] - boxes[:, 0]
    height_boxes = boxes[:, 3] - boxes[:, 1]

    plt.scatter(width_boxes, height_boxes)
    plt.show()

    # np.histogram(width_boxes, bins=100)
    plt.hist(height_boxes, bins=100)
    plt.show()

    plt.hist(width_boxes, bins=100)
    plt.show()

    bboxes = torch.tensor(boxes, dtype=torch.float)

    # k = 9
    # rows = bboxes.size()[0]
    # cluster_indxs = np.random.choice(rows, k, replace=False)
    # anchors = bboxes[cluster_indxs].clone()

    N_CLUSTERS = 9
    anchors = KMeans_clustering_anchor_boxes(
        bboxes,
        k=N_CLUSTERS,
        stop_iter=2
    )
    print(anchors)
