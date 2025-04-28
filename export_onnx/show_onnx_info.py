import onnx
 
def show_onnx_info(model_path):
    # 加载ONNX模型
    # model_path = "/root/lsj/yolo11/runs/detect/train109/weights/best.onnx"
    model = onnx.load(model_path)
    
    # 获取模型的元数据属性
    metadata_props = {prop.key: prop.value for prop in model.metadata_props}
    
    # 输出模型的元数据属性
    print(f"Model metadata properties: {metadata_props}")
    
    # 获取IR版本
    # model.ir_version = ir_version
    print(f"IR version of the model: {model.ir_version}")
    
def update_onnx_ir_version(model_path,ir_version):
    model = onnx.load(model_path)
    model.ir_version = ir_version
    onnx.save_model(model, model_path)
'''
Model metadata properties: 
{'description': 'Ultralytics YOLO11n model trained on /root/lsj/yolo11/run_det.yaml', 
'author': 'Ultralytics', 
'date': '2025-02-19T15:31:43.281749', 
'version': '1.0.0', 
'license': 'AGPL-3.0 License (https://ultralytics.com/license)', 
'docs': 'https://docs.ultralytics.com', 
'stride': '32', 'task': 'detect', 'batch': '1', 
'imgsz': '[3840, 3840]', 
'names': "{0: '0', 1: '1', 2: '2', 3: '3', 
        4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}", 
'Company_name': 'dinnar', 
'export_date': '2025-02-19-15-31-41'}
IR version of the model: 10
'''