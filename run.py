from ultralytics import YOLO



def tune_screen_2000_v11(gpu=[0,1]):
    data = '/root/lsj/ultralytics-main/yolov8-datasets-screen-slice.yaml'
    data = '/root/lsj/ultralytics-main/yolov8-datasets-slice-1709-785.yaml'
    ckp = 'yolo11s.pt'
    
    model = YOLO(ckp)
    model.tune(data=data,
               batch=16, 
               imgsz= 1024,
               device=gpu,
               epochs=10, iterations=100, optimizer="AdamW", 
               plots=False, save=False, val=False)
    return


def tune_2000_cls_v11(gpu=[0,1]):
    data = '/root/datasets/yolo0703-screen-slice-cls'
    ckp = 'yolo11s-cls.pt'
    model = YOLO(ckp)
    model.tune(data=data,
               batch=16, 
               imgsz= 1024,
               device=gpu,
               epochs=10, iterations=100, optimizer="AdamW", 
               plots=False, save=False, val=False)
    pass

def train_2000_v11_directly(gpu=[0,1]):
    data = '/root/lsj/ultralytics-main/yolov8-datasets-screen-slice.yaml'
    data = '/root/lsj/ultralytics-main/yolov8-datasets-slice-1709-785.yaml'
    ckp = 'yolo11s.pt'
    model = YOLO(ckp)
    model.train(data=data, 
                device=gpu,
                batch=16,imgsz=1024,
                epochs=500,
                )

def train_2000_v11(gpu=[0,1]):
    data = '/root/lsj/ultralytics-main/yolov8-datasets-screen-slice.yaml'
    data = '/root/lsj/ultralytics-main/yolov8-datasets-slice-1709-785.yaml'
    ckp = 'yolo11s.pt'
    model = YOLO(ckp)
    model.train(data=data, 
                device=gpu,
                batch=8,imgsz=2000,
                epochs=500,
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
                )

def train_2000_cls_v11(gpu=[0,1]):
    data = '/root/datasets/yolo0703-screen-slice-cls'
    ckp = 'yolo11s-cls.pt'
    model = YOLO(ckp)
    model.train(data=data, 
                device=gpu,
                batch=8,imgsz=2000,
                epochs=500,
                optimizer="AdamW",
                lr0= 0.01055,lrf= 0.00893,
                momentum= 0.95573, weight_decay= 0.00044, warmup_epochs= 4.16745,
                warmup_momentum= 0.88576,
                box= 9.54871,cls= 0.43926,dfl= 1.13169,
                hsv_h= 0.00531,hsv_s= 0.59788,hsv_v= 0.39349,
                degrees= 0.0,
                translate= 0.12023,scale= 0.38433,shear= 0.0,
                perspective= 0.0,
                flipud= 0.0,fliplr= 0.38197,
                bgr= 0.0,mosaic= 0.51095,mixup= 0.0,copy_paste= 0.0,
                )



if __name__=='__main__':
    ############### tune ##############
    # tune_screen_2000_v11(gpu=[2])
    # tune_2000_cls_v11(gpu=[2])
    
    ############## train ###############
    # train_2000_v11(gpu=[3])
    train_2000_v11_directly(gpu=[2])
    # train_2000_cls_v11(gpu=[3])