import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from ultralytics.cfg import TASK2DATA
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    __version__,
    colorstr,
)
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
def export_formats():
    """Ultralytics YOLO export formats."""
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False],
        ["TensorFlow.js", "tfjs", "_web_model", True, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
        ["NCNN", "ncnn", "_ncnn_model", True, True],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU"], zip(*x)))



@smart_inference_mode()
def export_like_yolo(self, model=None,extra_metadata={}) -> str:
    """Returns list of exported files/dirs after running callbacks."""
    self.run_callbacks("on_export_start")
    t = time.time()
    fmt = self.args.format.lower()  # to lowercase
    if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
        fmt = "engine"
    fmts = tuple(export_formats()["Argument"][1:])  # available export formats

    flags = [x == fmt for x in fmts]
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, ncnn = flags  # export booleans
    is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

    # Device
    if fmt == "engine" and self.args.device is None:
        LOGGER.warning("WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0")
        self.args.device = "0"
    self.device = select_device("cpu" if self.args.device is None else self.args.device)

    # Checks
    model.names = check_class_names(model.names)
    if self.args.half and self.args.int8:
        LOGGER.warning("WARNING ⚠️ half=True and int8=True are mutually exclusive, setting half=False.")
        self.args.half = False
    if self.args.half and onnx and self.device.type == "cpu":
        LOGGER.warning("WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0")
        self.args.half = False
        assert not self.args.dynamic, "half=True not compatible with dynamic=True, i.e. use only one."
    self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size stride=tensor([ 8., 16., 32.])
    if self.args.int8 and engine:
        self.args.dynamic = True  # enforce dynamic to export TensorRT INT8
    if self.args.optimize:
        assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
        assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"

    if self.args.int8 and not self.args.data:
        self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
        LOGGER.warning(
            "WARNING ⚠️ INT8 export requires a missing 'data' arg for calibration. "
            f"Using default 'data={self.args.data}'."
        )
    # Input
    im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)
    file = Path(
        getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
    )
    if file.suffix in {".yaml", ".yml"}:
        file = Path(file.name)

    # Update model
    model = deepcopy(model).to(self.device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for m in model.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
            m.dynamic = self.args.dynamic
            m.export = True
            m.format = self.args.format
            m.max_det = self.args.max_det
        elif isinstance(m, C2f) and not is_tf_format:
            # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
            m.forward = m.forward_split

    y = None
    for _ in range(2):
        y = model(im)  # dry runs torch.Size([1, 15, 302400])
    if self.args.half and onnx and self.device.type != "cpu":
        im, model = im.half(), model.half()  # to FP16

    # Filter warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

    # Assign
    self.im = im
    self.model = model
    self.file = file
    self.output_shape = (
        tuple(y.shape)
        if isinstance(y, torch.Tensor)
        else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
    )
    self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
    data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
    description = f'Ultralytics {self.pretty_name} model {f"trained on {data}" if data else ""}'
    self.metadata = {
        "description": description,
        "author": "Ultralytics",
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": int(max(model.stride)),
        "task": model.task,
        "batch": self.args.batch,
        "imgsz": self.imgsz,
        "names": model.names,
    }  # model metadata
    for k,v in extra_metadata.items():
        self.metadata[k] = v

    '''
    PyTorch: starting from 'runs/detect/train109/weights/best.pt' with 
    input shape (1, 3, 3840, 3840) BCHW and 
    output shape(s) (1, 15, 302400) (6.3 MB)
    '''
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
        f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)'
    )

    # Exports
    f = [""] * len(fmts)  # exported filenames
    if engine:  # TensorRT required before ONNX
        f[1], _ = self.export_engine()
    if onnx:  # ONNX
        f[2], _ = self.export_onnx()
    

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        f = str(Path(f[-1]))
        square = self.imgsz[0] == self.imgsz[1]
        s = (
            ""
            if square
            else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
            f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
        )
        imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
        predict_data = f"data={data}" if model.task == "segment" and fmt == "pb" else ""
        q = "int8" if self.args.int8 else "half" if self.args.half else ""  # quantization
        LOGGER.info(
            f'\nExport complete ({time.time() - t:.1f}s)'
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f'\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}'
            f'\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}'
            f'\nVisualize:       https://netron.app'
        )

    self.run_callbacks("on_export_end")
    return f  # return list of exported files/dirs