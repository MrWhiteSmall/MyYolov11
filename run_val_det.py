from ultralytics import YOLO,solutions
import cv2
import numpy as np
# Load a model
ckp = 'runs/detect/train109/weights/best.pt'

model = YOLO(ckp)  # load a custom model

imgpath = '/root/datasets/mvYOLOTP/images/train/T132C06A24CD00219_Up302-35-03_flipl2r.jpg'
# Predict with the model
results = model(imgpath)  # predict on an image