import os

import torch

from src.torch_implementation.model_retinanet import create_retinanet_model

IMAGE_SIZE = (1024, 1024)
onnx_model_path = 'detector.onnx'

if __name__ == '__main__':
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'
    import torch

    # Load retinanet
    # pth_path = "/path/to/retinanet.pth"
    retinanet = create_retinanet_model(
        score_threshold=0.2,
        iou_threshold=0.2,
        num_classes=2
    )
    retinanet.eval()

    # Image sizes
    original_image_size = (1024, 1024)

    # Normalize hack
    normalize_tmp = retinanet.transform.normalize
    retinanet_normalize = lambda x: normalize_tmp(x)
    retinanet.transform.normalize = lambda x: x

    # Resize hack
    resize_tmp = retinanet.transform.resize
    retinanet_resize = lambda x: resize_tmp(x, None)[0]
    retinanet.transform.resize = lambda x, y: (x, y)

    # Batch images hack
    # /!\ torchvision version dependent ???
    retinanet.transform.batch_images = lambda x, size_divisible: x[0].unsqueeze(0)
    # retinanet.transform.batch_images = lambda x: x[0].unsqueeze(0)


    # Generate dummy input
    def preprocess_image(img):
        result = retinanet_resize(retinanet_normalize(img)[0]).unsqueeze(0)
        return result


    dummy_input = torch.randn(1, 3, original_image_size[0], original_image_size[1])
    dummy_input = preprocess_image(dummy_input)
    image_size = tuple(dummy_input.shape[2:])
    print(dummy_input.shape)

    # Postprocess detections hack
    postprocess_detections_tmp = retinanet.postprocess_detections
    retinanet_postprocess_detections = lambda x: postprocess_detections_tmp(x["split_head_outputs"], x["split_anchors"],
                                                                            [image_size])
    retinanet.postprocess_detections = lambda x, y, z: {"split_head_outputs": x, "split_anchors": y}

    # Postprocess hack
    postprocess_tmp = retinanet.transform.postprocess
    retinanet_postprocess = lambda x: postprocess_tmp(x, [image_size], [original_image_size])
    retinanet.transform.postprocess = lambda x, y, z: x

    # ONNX export
    onnx_path = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\ONNX_tests\ONNX_models\retinanet.onnx"
    torch.onnx.export(
        retinanet,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=11,
        input_names=["images"],
    )