import torch
import torchvision

from ultralytics import YOLO
import onnx,torch,time
# Load a model
ckp = 'runs/detect/train109/weights/best.pt'
model = YOLO(ckp)  # load a custom trained model

device = 'cuda' if torch.cuda.is_available else 'cpu'

dummy_input = torch.randn(1, 3, 3840, 3840, device=device)
model.to(device)
# model.eval()
output = model(dummy_input)[0]
print("pytorch result:", torch.argmax(output))

# --------------------------导出onnx模型--------------------------
import torch.onnx
torch.onnx.export(model, dummy_input, './model.onnx', 
                  input_names=["input"], 
                  output_names=["output"], 
                  do_constant_folding=True, 
                  verbose=True, 
                  keep_initializers_as_inputs=True, 
                  opset_version=12, 
                  dynamic_axes={"input": {0: "nBatchSize"}, "output": {0: "nBatchSize"}})

# --------------------------验证onnx模型--------------------------
import onnx
import numpy as np
import onnxruntime as ort

model_onnx_path = './model.onnx'
# 验证模型的合法性
onnx_model = onnx.load(model_onnx_path)
onnx.checker.check_model(onnx_model)
# 创建ONNX运行时会话
ort_session = ort.InferenceSession(model_onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备输入数据
input_data = {
    'input': dummy_input.cpu().numpy()
}
# 运行推理
y_pred_onnx = ort_session.run(None, input_data)
np.argmax(y_pred_onnx[0])