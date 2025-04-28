# runs/detect/predict/labels
predict_root=runs/detect/predict13
data_root=/root/datasets/mvTP-FUMO-mixed-Aug/validate

python /root/lsj/yolo11/run_validate/tool_for_judge_det.py \
--predict_dir $predict_root/labels \
--target_label_dir $data_root/labels \
--ori_img_dir $data_root/images \
--save_img_dir $data_root/val1 \
--conf 0.2