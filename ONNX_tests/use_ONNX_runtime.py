import time
import typing

import numpy as np
import onnx
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ONNX_tests.export_ONNX_format import MIN_CONFIDENCE, MIN_IOU_THRESHOLD, class_mapping
from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.model_retinanet import create_retinanet_model, collate_fn

IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = 1
MIN_CONFIDENCE: typing.Final[float] = 0.2


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


transform = A.Compose([
    # A.Normalize(),
    A.RandomCrop(*IMAGE_SIZE),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.5))

train_dataset = PascalVOCDataset(
    data_folder=r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\dataset\test",
    split='train',
    single_cls=True,
    transform=transform)

data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # collate_fn=train_dataset.collate_fn
    collate_fn=collate_fn
)

if __name__ == '__main__':
    import onnxruntime

    onnx_model_file = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\ONNX_tests\ONNX_models\retinanet.onnx"

    model = onnx.load(onnx_model_file)
    # inits = model.graph.initializer
    # for init in inits:
    #     print(init.name)

    ort_session = onnxruntime.InferenceSession(onnx_model_file, providers=[
                                                                           'CPUExecutionProvider'])
    print(f'Used ONNX device: {onnxruntime.get_device()}')

    for i in range(5):

        # x = torch.randn(1, 3, 1024, 1024, requires_grad=True)
        x = torch.unsqueeze(next(iter(data_loader))[0][0], dim=0)

        # compute ONNX Runtime output prediction
        start_prediction_onnx = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        print(f"Time taken to run ONNX: {time.time() - start_prediction_onnx}")

        retinanet = create_retinanet_model(num_classes=len(class_mapping),
                                           use_pretrained_weights=True,
                                           score_threshold=MIN_IOU_THRESHOLD,
                                           iou_threshold=MIN_CONFIDENCE,
                                           unfrozen_layers=3,
                                           mean_values=(0.9629258011853685, 1.1043921727662964, 0.9835339608076883),
                                           std_values=(0.08148765554920795, 0.10545005065566, 0.13757230267160245)
                                           )
        retinanet.eval()

        start_prediction_torch = time.time()
        torch_out = retinanet(x)
        print(f"Time taken to run Pytorch: {time.time() - start_prediction_torch}")



        # compare ONNX Runtime and PyTorch results
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        torch_predictions = np.hstack((to_numpy(torch_out[0]['boxes']),
                                       np.expand_dims(to_numpy(torch_out[0]['labels']), axis=1),
                                       np.expand_dims(to_numpy(torch_out[0]['scores']), axis=1)))


        ort_predictions = np.hstack((ort_outs[0],
                                    np.expand_dims(ort_outs[2], axis=1),
                                    np.expand_dims(ort_outs[1], axis=1)))

        torch_predictions = torch_predictions[torch_predictions[:, -1] > MIN_CONFIDENCE]
        ort_predictions = ort_predictions[ort_predictions[:, -1] > MIN_CONFIDENCE]

        np.testing.assert_allclose(torch_predictions[torch_predictions[:, -1].argsort()],
                                   ort_predictions[ort_predictions[:, -1].argsort()], rtol=1e-03, atol=1e-05)



        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
