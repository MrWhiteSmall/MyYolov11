# /root/datasets/mvYOLOTP

from ultralytics import YOLO

if __name__=='__main__':
    # Load a model
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    data = 'run_seg.yaml'
    # Train the model
    gpu=[2]
    results = model.train(batch=1, 
                          data=data,
                        imgsz= 2048,
                        device=gpu,
                        epochs=450, )