# runs/detect/predict/labels
predict_root=runs/detect/predict13
# data_root=/root/datasets/mvTP-SIMO-Light-Aug/validate
data_root=/root/datasets/mvTP-FUMO-mixed-Aug/validate

python /root/lsj/yolo11/run_validate/tool_for_judge_confuse_type.py \
--data_root $data_root \
--validate_img_dir $data_root/images \
--validate_label_dir $data_root/labels \
--pred_label_dir $predict_root/labels \
--conf_threshold 0.2