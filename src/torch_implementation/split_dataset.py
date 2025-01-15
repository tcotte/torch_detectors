import os
import shutil

import imutils.paths
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    root = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras"
    path_annotations = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset_pascal_voc\Evry_dataset_2024_pascal_voc\VOCdevkit\VOC\Annotations"

    X = list(imutils.paths.list_images(r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset_pascal_voc\Evry_dataset_2024_pascal_voc\VOCdevkit\VOC\JPEGImages"))

    y = [os.path.join(path_annotations, i) for i in os.listdir(path_annotations)]

    assert len(X) == len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    for set_ in ["train", "test"]:
        for j in ["y", "X"]:
            folder_name = "Annotations" if j == "y" else "JPEGImages"
            dst_folder = os.path.join(root, "dataset", set_, folder_name)
            os.makedirs(dst_folder, exist_ok=True)
            for i in globals()[f'{j}_{set_}']:
                shutil.copy(i, os.path.join(dst_folder, os.path.basename(i)))
