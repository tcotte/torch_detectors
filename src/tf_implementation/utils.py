from typing import Union

# from tensorflow.python.data.ops.take_op import _TakeDataset, SplitDataset


def count_annotations_in_split(split: Union['_TakeDataset', 'SplitDataset'], class_mapping: dict):
    d = {}
    for key in class_mapping.keys():
        c = 0
        for element in split:
            c += element[-2].numpy().tolist().count(key)
        d[class_mapping[key]] = c

    return d