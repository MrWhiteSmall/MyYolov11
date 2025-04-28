imgsize=2048

python /root/lsj/yolo11/export_onnx/export_full_test_slice.py \
--model /root/lsj/yolo11/runs/detect/train144/weights/best.onnx \
--input_img /root/datasets/mvYOLO-Det-TP-Aug/2025-2-20/T132C06A24CD00219_Up302-35-03.bmp \
--input_dir /root/datasets/mvYOLO-Det-TP-Aug/validate_whole/images/ \
--slice_h $imgsize \
--slice_w $imgsize \
--conf_thres 0.2 \
--iou_thres 0.5