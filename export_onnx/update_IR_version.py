import onnx
 
# 加载ONNX模型
model_path = "/root/lsj/yolo11/runs/detect/train109/weights/best.onnx"
model = onnx.load(model_path)


model.ir_version = 7
onnx.save_model(model, model_path)