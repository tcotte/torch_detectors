import os

import torch

from src.torch_implementation.model_retinanet import create_retinanet_model



# https://github.com/pytorch/vision/issues/4395#issuecomment-1086658634
IMAGE_SIZE = (1024, 1024)
onnx_model_path = 'detector.onnx'
OUTPUT_PATH = r"C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\ONNX_tests\ONNX_models\retinanet.onnx"
MIN_CONFIDENCE = 0.5
MIN_IOU_THRESHOLD = 0.01

class_mapping = {
    0: 'ft'
}


if __name__ == '__main__':
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'
    import torch

    # Load retinanet
    # pth_path = "/path/to/retinanet.pth"
    retinanet = create_retinanet_model(num_classes=len(class_mapping),
                                   use_pretrained_weights=True,
                                   score_threshold=MIN_IOU_THRESHOLD,
                                   iou_threshold=MIN_CONFIDENCE,
                                   unfrozen_layers=3,
                                   mean_values=(0.9629258011853685, 1.1043921727662964, 0.9835339608076883),
                                   std_values=(0.08148765554920795, 0.10545005065566, 0.13757230267160245)
                                   )

    retinanet.eval()
    retinanet.cuda()

    # Image sizes
    original_image_size = IMAGE_SIZE

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
    # dummy_input = preprocess_image(dummy_input)
    image_size = tuple(dummy_input.shape[2:])
    dummy_input = dummy_input.cuda()

    # Postprocess detections hack
    # postprocess_detections_tmp = retinanet.postprocess_detections
    # retinanet_postprocess_detections = lambda x: postprocess_detections_tmp(x["split_head_outputs"], x["split_anchors"],
    #                                                                         [image_size])
    # retinanet.postprocess_detections = lambda x, y, z: {"split_head_outputs": x, "split_anchors": y}
    #
    # # Postprocess hack
    # postprocess_tmp = retinanet.transform.postprocess
    # retinanet_postprocess = lambda x: postprocess_tmp(x, [image_size], [original_image_size])
    # retinanet.transform.postprocess = lambda x, y, z: x

    # ONNX export
    torch.onnx.export(
        retinanet,
        dummy_input,
        OUTPUT_PATH,
        verbose=False,
        opset_version=11,
        input_names=["images"],
        output_names=['boxes', 'scores', 'labels'],
    )
