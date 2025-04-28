# (4800,4300)
data=/root/lsj/yolo11/run_det_light_slice.yaml
ckp=yolo11s.pt

# python run_det.py \
#     --gpus 2 --epoch 500 \
#     --bz 1 --imgsize 4300 \
#     --data $data \
#     --ckp $ckp

# 18 GB
python run_det.py \
    --gpus 2 --epoch 500 \
    --bz 4 --imgsize 1024 \
    --data $data \
    --ckp $ckp