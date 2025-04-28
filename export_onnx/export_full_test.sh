python /root/lsj/yolo11/export_onnx/export_full_test.py \
--model /root/lsj/yolo11/runs/detect/train109/weights/best.onnx \
--img /root/datasets/mvYOLO-Det-TP-Aug/2025-2-20/T132C06A24CD00219_Up302-35-03.bmp \
--conf-thres 0.75 \
--iou-thres 0.5