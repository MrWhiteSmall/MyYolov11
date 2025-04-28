from ultralytics import YOLO
import time
# Load a model
ckp = 'runs/detect/train109/weights/best.pt'
onnx_path = 'runs/detect/train109/weights/best.onnx'
model = YOLO(ckp)  # load a custom trained model

date=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# Metadata
extra_metadata = {
     'Company_name':'dinnar',
     'version':'1.0.0',
     'export_date':date
     }
# 导出模型，设置多种参数
export_file_list  = model.export(
    extra_metadata=extra_metadata,
    format="onnx",      # 导出格式为 ONNX
    imgsz=(3840, 3840),   # 设置输入图像的尺寸
    keras=False,        # 不导出为 Keras 格式
    optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
    half=False,         # 不启用 FP16 量化
    int8=False,         # 不启用 INT8 量化
    dynamic=False,      # 不启用动态输入尺寸
    simplify=True,      # 简化 ONNX 模型
    opset=None,         # 使用最新的 opset 版本
    workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
    nms=False,          # 不添加 NMS（非极大值抑制）
    batch=1,            # 指定批处理大小
    device="0"        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
)

print(export_file_list)