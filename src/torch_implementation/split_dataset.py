import argparse
import os
import shutil

import imutils.paths
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    prog='RetinaNet_Trainer',
    description='The aim of this program is to split dataset into test and train sets',
    epilog='------- SGS France - Operational Innovation -------')
parser.add_argument('--destination', type=str, required=True,
                    help='Path where the split dataset will be saved')
parser.add_argument('--path_annotations', type=str, required=True,
                    help='Path of folder containing annotations')
parser.add_argument('--path_images', type=str, required=True,
                    help='Path of folder containing images')
parser.add_argument('--test_size', type=float, default=0.15, required=False,
                    help='Validation test size')
parser.add_argument('--random_state', type=int, default=42, required=False,
                    help='Seed used to split datasets')
args = parser.parse_args()


if __name__ == "__main__":
    destination_folder: str = args.destination
    path_annotations: str = args.path_annotations
    X: list[str] = list(imutils.paths.list_images(args.path_images))

    y: list[str] = [os.path.join(path_annotations, i) for i in os.listdir(path_annotations)]

    assert len(X) == len(y)

    X_train, X_test, y_train, y_test = train_test_split(sorted(X), sorted(y), test_size=args.test_size,
                                                        random_state=args.random_state)

    for set_ in ["train", "test"]:
        for j in ["y", "X"]:
            folder_name = "Annotations" if j == "y" else "JPEGImages"
            dst_folder = os.path.join(destination_folder, "dataset", set_, folder_name)
            os.makedirs(dst_folder, exist_ok=True)
            for i in globals()[f'{j}_{set_}']:
                shutil.copy(i, os.path.join(dst_folder, os.path.basename(i)))
