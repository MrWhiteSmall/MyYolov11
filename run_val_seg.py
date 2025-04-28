from ultralytics import YOLO,solutions
import cv2
import numpy as np
# Load a model
# model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("/root/lsj/yolo11/runs/segment/train11/weights/best.pt")  # load a custom model

imgpath = '/root/datasets/mvYOLOTP/images/train/T132C06A24CD00219_Up302-35-03_flipl2r.jpg'
# Predict with the model
results = model(imgpath)  # predict on an image

# Run batched inference on a list of images
# results = model([imgpath], stream=True)  # return a generator of Results objects

# Run inference on 'bus.jpg' with arguments , imgsz=2048
results = model.predict(imgpath, save=True, conf=0.1)

for pixel_xy in results[0].masks.xy[1]:
    points = np.array(pixel_xy, np.int32)
    input_image = cv2.imread('bus.jpg')
    cv2.drawContours(input_image, [points], -1, (0, 255, 0), 2)
    cv2.imwrite('output.jpg', input_image)

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
    
    # 加载原始图像
    image = cv2.imread(imgpath)
    
    # 将每个掩码与原始图像进行叠加
    for mask in masks:
        # 将掩码转换为二值图像（0 或 255）
        mask = mask.cpu().numpy()  # 将掩码转换为numpy数组
        mask = np.uint8(mask * 255)  # 转换为0或255

        # 将掩码应用到图像上，使用颜色标记掩码
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # 使用Jet色图着色掩码
        masked_image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)  # 叠加原图和掩码
        
    # 保存叠加后的图像
    output_path = "./segmented_with_mask.jpg"
    cv2.imwrite(output_path, masked_image)
    
    
