import os
os.getcwd()


from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

yolov11_model_path = "/root/lsj/yolo11/runs/detect/train109/weights/best.pt"
img_path = '/root/datasets/mvYOLO-Det-TP-Aug/2025-2-20/T132C06A24CD00219_Up302-35-03.bmp'
result_save_dir = './SAHI_res'

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov11',
    # YOLOv8模型的路径
    model_path=yolov11_model_path,
    # YOLOv8模型的路径
    confidence_threshold=0.3,
    # 设备类型。
    # 如果您的计算机配备 NVIDIA GPU，则可以通过将 'device' 标志更改为'cuda:0'来启用 CUDA 加速；否则，将其保留为'cpu'
    device="cuda:2", # or 'cuda:0'
)
'''
Args:
    image: str or np.ndarray
        Location of image or numpy image matrix to slice
    detection_model: model.DetectionModel
    slice_height: int
        Height of each slice.  Defaults to ``None``.
    slice_width: int
        Width of each slice.  Defaults to ``None``.
    overlap_height_ratio: float
        Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
        of size 512 yields an overlap of 102 pixels).
        Default to ``0.2``.
    overlap_width_ratio: float
        Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
        of size 512 yields an overlap of 102 pixels).
        Default to ``0.2``.
    perform_standard_pred: bool
        Perform a standard prediction on top of sliced predictions to increase large object
        detection accuracy. Default: True.
    postprocess_type: str
        Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
        Options are 'NMM', 'GREEDYNMM' or 'NMS'. Default is 'GREEDYNMM'.
    postprocess_match_metric: str
        Metric to be used during object prediction matching after sliced prediction.
        'IOU' for intersection over union, 'IOS' for intersection over smaller area.
    postprocess_match_threshold: float
        Sliced predictions having higher iou than postprocess_match_threshold will be
        postprocessed after sliced prediction.
    postprocess_class_agnostic: bool
        If True, postprocess will ignore category ids.
    verbose: int
        0: no print
        1: print number of slices (default)
        2: print number of slices and slice/prediction durations
    merge_buffer_length: int
        The length of buffer for slices to be used during sliced prediction, which is suitable for low memory.
        It may affect the AP if it is specified. The higher the amount, the closer results to the non-buffered.
        scenario. See [the discussion](https://github.com/obss/sahi/pull/445).
    auto_slice_resolution: bool
        if slice parameters (slice_height, slice_width) are not given,
        it enables automatically calculate these params from image resolution and orientation.
    slice_export_prefix: str
        Prefix for the exported slices. Defaults to None.
    slice_dir: str
        Directory to save the slices. Defaults to None.
'''
# Performing prediction on 6 slices.
result = get_sliced_prediction(
    img_path,
    detection_model,
    slice_height = 3840,
    slice_width = 3840,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)
'''
export_dir: directory for resulting visualization to be exported
text_size: size of the category name over box
rect_th: rectangle thickness
hide_labels: hide labels
hide_conf: hide confidence
file_name: saving name
'''
result.export_visuals(export_dir=result_save_dir,
                      export_format='jpg')

Image(f"{result_save_dir}/prediction_visual.jpg")