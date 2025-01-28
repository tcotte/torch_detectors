import time
import typing
import os
import torch
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ONNX_tests.export_ONNX_format import IMAGE_SIZE, class_mapping, MIN_IOU_THRESHOLD, MIN_CONFIDENCE
from src.torch_implementation.dataset import PascalVOCDataset
from src.torch_implementation.model_retinanet import collate_fn, build_retinanet_model

'''
We have to get a coherent set of libraries (Pytorch / CUDNN / onnxruntime) to run ONNX with CUDA:
See the link below for more details:
https://github.com/microsoft/onnxruntime/issues/22198#issuecomment-2376010703

Our environment:
CUDA: 12.1
cuDNN: 9.1.0
onnxruntime: 1.19.2

Documentation ONNX runtime on CUDA:
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html 
'''
BATCH_SIZE = 1


def get_CUDA_and_CUDNN_versions() -> dict:
    cudnn = torch.backends.cudnn.version()
    cudnn_major = cudnn // 10000
    cudnn = cudnn % 1000
    cudnn_minor = cudnn // 100
    cudnn_patch = cudnn % 100

    return {
        'CUDA': torch.version.cuda,
        'CUDNN': '.'.join([str(cudnn_major), str(cudnn_minor), str(cudnn_patch)])
    }


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

ONNX_MODEL_PATH: typing.Final[str] = (r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\ONNX_tests\ONNX_models'
                                      r'\retinanet.onnx')

if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True
    # torch.jit.optimized_execution(True)


    os.environ['CUDA_CACHE_DISABLE'] = '0'
    # torch._C._jit_set_profiling_mode(False)  # Fixes initial delay

    retinanet = build_retinanet_model(num_classes=len(class_mapping),
                                      use_COCO_pretrained_weights=True,
                                      score_threshold=MIN_IOU_THRESHOLD,
                                      iou_threshold=MIN_CONFIDENCE,
                                      unfrozen_layers=3,
                                      mean_values=(0.9629258011853685, 1.1043921727662964, 0.9835339608076883),
                                      std_values=(0.08148765554920795, 0.10545005065566, 0.13757230267160245)
                                      )
    retinanet.eval()
    retinanet.cuda()

    providers = ["CUDAExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=sess_options, providers=providers)
    print("Available Providers:", ort.get_available_providers())
    print(sess.get_providers())

    io_binding = sess.io_binding()

    # Pass gpu_graph_id to RunOptions through RunConfigs
    ro = ort.RunOptions()

    # compute ONNX Runtime output prediction
    for i in range(5):
        x = torch.unsqueeze(next(iter(data_loader))[0][0], dim=0)

        start_prediction_onnx = time.time()
        ort_inputs = {sess.get_inputs()[0].name: to_numpy(x)}
        ort_outs = sess.run(None, ort_inputs)
        print(f"Time taken to run ONNX: {time.time() - start_prediction_onnx}")

        x = x.cuda()

        start_prediction_torch = time.time()
        torch_out = retinanet(x)
        print(f"Time taken to run Pytorch: {time.time() - start_prediction_torch}")

    # x_ortvalue = ort.OrtValue.ortvalue_from_numpy(x.numpy(), 'cuda', 0)
    #
    # # Bind the input and output
    # io_binding.bind_ortvalue_input('X', x_ortvalue)
    #
    #
    # # One regular run for the necessary memory allocation and cuda graph capturing
    # sess.run_with_iobinding(io_binding, ro)

    # providers = [("CUDAExecutionProvider", {'enable_cuda_graph': True})]
