# onnxruntime-gpu：ONNX Runtime 的 GPU 加速版，用来在 NVIDIA GPU 上执行推理。
# py3nvml：用来监控 GPU 的 Python 库，可以获取 GPU 的内存使用情况和占用。

onnx_model_path = '/root/lsj/yolo11/runs/detect/train109/weights/best.onnx'
img_path = '/root/datasets/mvYOLO-Det-TP-Aug/images/train/T132C06A24CD00279_Up302-36-04_horizontal.jpg'  # 更改为你的图片路径

import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import pynvml   # 用于获取 GPU 占用情况


# 加载图片并进行预处理
img = Image.open(img_path)

transform = transforms.Compose([
    transforms.Resize((3840, 3840)),  # Resize 到 640x640
    transforms.ToTensor(),          # 转为张量
])

input_tensor = transform(img).unsqueeze(0).numpy()  # 维度 (1, 3, 640, 640)


# 初始化 py3nvml
pynvml.nvmlInit()

# 加载 ONNX 模型
# onnx_model_path = 'yolov5.onnx'

# 加载 ONNX Runtime 会话，使用 GPU 加速
# Available providers: 'TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider'
'''
2025-02-18 16:58:35.116566148 
[E:onnxruntime:Default, provider_bridge_ort.cc:1848 TryGetProviderInfo_TensorRT] 
/onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1539 
    onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : 
    Failed to load library libonnxruntime_providers_tensorrt.so with error: libcublas.so.12: 
    cannot open shared object file: No such file or directory
'''
sess = ort.InferenceSession(onnx_model_path, providers=['TensorrtExecutionProvider'])


# 打印开始前的 GPU 状态
def print_gpu_memory():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 获取第 0 个 GPU
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory Usage: {memory_info.used / 1024**2:.2f} MB / {memory_info.total / 1024**2:.2f} MB")

print_gpu_memory()

# 记录推理开始时间
start_time = time.time()

# 执行推理
outputs = sess.run(None, {'images': input_tensor})  # 输入 'input' 是 ONNX 模型定义的输入名称

# 记录推理结束时间
end_time = time.time()

# 打印推理时间
inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

# 打印 GPU 使用情况
print_gpu_memory()

'''
GPU Memory Usage: 624.06 MB / 40960.00 MB
Inference Time: 1.9463 seconds
GPU Memory Usage: 624.06 MB / 40960.00 MB
'''
# 输出结果
print("Inference Output:")
print(outputs.shape)
