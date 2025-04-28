from ultralytics.models import RTDETR
 
if __name__ == '__main__':
    model = RTDETR(model = 'rtdetr-detection-l.pt')
    data = r'/root/lsj/yolo11/run_det_slice.yaml'
    data = r'/root/lsj/yolo11/run_det_light_slice.yaml'
    gpu = 2
    '''
    device=args.gpus,
            batch=args.bz,
            imgsz=args.imgsize,
            epochs=args.epoch,
            
            
            optimizer="AdamW",
            lr0= 0.0033,lrf= 0.00505,
            momentum= 0.86103, weight_decay= 0.0005, warmup_epochs= 2.41019,
            warmup_momentum= 0.42804,
            box= 7.7261,cls= 0.43791,dfl= 0.90948,
            hsv_h= 0.0295,hsv_s= 0.60304,hsv_v= 0.44799,
            degrees= 0.0,
            translate= 0.05768,scale= 0.07865,shear= 0.0,
            perspective= 0.0,
            flipud= 0.0,fliplr= 0.50592,
            bgr= 0.0,mosaic= 0.93816,mixup= 0.0,copy_paste= 0.0,
    '''
    model.train(pretrained=True, 
                data=data,
                epochs=500,
                batch=1, 
                device=gpu,
                imgsz=3200, 
                
                workers=8,
                project='runs/train',
                name='exp',
                )
 