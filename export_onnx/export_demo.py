from ultralytics import YOLO
import onnx,torch,time
# Load a model
ckp = 'runs/detect/train109/weights/best.pt'
model = YOLO(ckp)  # load a custom trained model

# 假设你已经加载了训练好的 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov11n')

# 导出 ONNX 模型
onnx_path = 'yolov11_with_version.onnx'
input_tensor = torch.randn(1, 3, 3840, 3840)  # 批次大小为1，3个通道，640x640的图像
torch.onnx.export(model,               # 要导出的模型
                  input_tensor,                # 模型的输入示例（例如，Tensor 或一组输入）
                  onnx_path,                   # 导出的 ONNX 文件路径
                  export_params=True,  # 是否导出模型的参数
                  opset_version=12,    # ONNX 操作集版本
                  do_constant_folding=True,  # 是否进行常量折叠优化
                  input_names=['input'],    # 输入张量的名称
                  output_names=['output'],  # 输出张量的名称
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # 动态轴
                  verbose=False)        # 是否打印导出过程中的详细信息


# 加载模型并添加版本信息
onnx_model = onnx.load(onnx_path)
date=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# Metadata
metadata = {'stride': int(max(model.stride)), 
     'names': model.names,
     'Company_name':'dinnar',
     'version':'1.0.0',
     'export_date':date}
onnx_model.metadata = metadata
onnx.save(onnx_model, onnx_path)
