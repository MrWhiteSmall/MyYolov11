arg1=$1

data=/root/lsj/yolo11/run_det_slice.yaml
ckp=yolo11s.pt
imgsz=1024

if [ "$arg1" -eq 0 ]; then  # runs/detect/train107
    python run_det.py \
    --gpus 0 --epoch 500 \
    --bz 2 --imgsize 2560 \
    --data /root/lsj/yolo11/run_det.yaml
elif [ "$arg1" -eq 1 ]; then # runs/detect/train108
    python run_det.py \
    --gpus 1 --epoch 500 \
    --bz 2 --imgsize 3200 \
    --data /root/lsj/yolo11/run_det.yaml
elif [ "$arg1" == 2 ]; then # runs/detect/train109
    python run_det.py \
    --gpus 2 --epoch 500 \
    --bz 1 --imgsize 3840 \
    --data /root/lsj/yolo11/run_det.yaml
elif [ "$arg1" == 3 ]; then # Results saved to runs/detect/train112
    python run_det.py \
    --gpus 1 --epoch 500 \
    --bz 16 --imgsize $imgsz \
    --data $data \
    --ckp $ckp
else
    echo "no this command"
fi