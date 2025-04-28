import argparse
import cv2,math,time
import numpy as np
import onnxruntime as ort

CLASS_NAMES = {i: f"CLASS_{i}" for i in range(12)}  # 示例类别名
# COLORS = {
#     "red": (255, 0, 0),
#     "green": (0, 255, 0),
#     "blue": (0, 0, 255),
#     "yellow": (255, 255, 0),
#     "cyan": (0, 255, 255),
#     "magenta": (255, 0, 255),
#     "orange": (255, 165, 0),
#     "purple": (128, 0, 128),
#     "pink": (255, 192, 203),
#     "brown": (165, 42, 42),
#     "gray": (128, 128, 128),
#     "lightgray": (211, 211, 211),
#     "darkgray": (169, 169, 169),
#     "white": (255, 255, 255),
#     "black": (0, 0, 0),
#     "lime": (0, 255, 0),
#     "teal": (0, 128, 128),
#     "navy": (0, 0, 128),
#     "maroon": (128, 0, 0),
#     "olive": (128, 128, 0)
# }

class YOLO11:
    def __init__(self, model_path):
        self.model_path = model_path
        
        # 初始化模型
        self.session = ort.InferenceSession(model_path)
        self.model_inputs = self.session.get_inputs()[0]
        self.target_size = (self.model_inputs.shape[2], self.model_inputs.shape[3])
        self.target_h,self.target_w = self.target_size
        
        self.get_n = lambda length,slice_width:math.ceil(length/slice_width)
        self.get_overlap = lambda slice_width,origin_size,n:int((slice_width*n-origin_size)/(n-1)) if n>1 else 0
        
        # 加载类别名称
        self.classes = CLASS_NAMES
        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def generate_slices(self, img, slice_h, slice_w, overlap_h,overlap_w):
        """生成带重叠的图像切片"""
        h, w = img.shape[:2]
        step_h = slice_h - overlap_h
        step_w = slice_w - overlap_w
        slices = []
        
        for y in range(0, h - slice_h + 1, step_h):
            y_end = y + slice_h
            if y_end > h: y_end = h
            for x in range(0, w - slice_w + 1, step_w):
                x_end = x + slice_w
                if x_end > w: x_end = w
                slices.append({
                    'img': img[y:y_end, x:x_end],
                    'x_start': x,
                    'y_start': y,
                    'width': x_end - x,
                    'height': y_end - y
                })
        return slices

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114),scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        print(f"Original image shape: {shape}")
 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
 
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)
 
        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
 
        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2
 
        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
 
        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh - 0.1)) , int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) , int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=color)
        print(f"Final letterboxed image shape: {img.shape}")
 
        return img, (r, r), (dw, dh)
 
 

    def preprocess_slice(self, slice_img):
        
        # 使用 OpenCV 读取输入图像
        img = slice_img
        # 获取输入图像的高度和宽度
        img_height, img_width = img.shape[:2]
 
        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, ratio, (dw, dh) = \
            self.letterbox(img, new_shape=(self.target_w, self.target_h))
 
        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0
 
        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
 
        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
 
        return image_data,ratio,dw,dh

    def detect(self, input_img,
               slice_h,slice_w,
               conf_thres,iou_thres,
               ):
        input_img = cv2.imread(input_img)
        h,w,_ = input_img.shape
        
        self.confidence = conf_thres
        self.iou = iou_thres
        self.slice_h = slice_h
        self.slice_w = slice_w
        overlap_h = self.get_overlap(slice_h,h,self.get_n(h,slice_h))
        overlap_w = self.get_overlap(slice_w,w,self.get_n(w,slice_w))
        # 生成切片
        self.slices = self.generate_slices(input_img, 
                                           slice_h, slice_w, 
                                           overlap_h,overlap_w)
        print('切片数量::',len(self.slices))
        
        """执行完整检测流程"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        ''' time record '''
        record_time_slice_onnx=[]
        record_time_slice     =[]
        record_time_all       =[]
        
        
        start_loop_time = time.time()
        for slice_info in self.slices:
            start_time = time.time()
            # 预处理
            input_data,ratio,dw,dh = self.preprocess_slice(slice_info['img'])
            # input_data = np.transpose(input_data, (2, 0, 1))
            # input_data = np.expand_dims(input_data, axis=0)
            start_onnx_time = time.time()
            # 推理  type(outputs)=list  outputs[0].shape=1*rows*data
            outputs = self.session.run(None, {self.model_inputs.name: input_data})
            # print('单张 onnx总用时',time.time()-start_onnx_time)
            record_time_slice_onnx.append(time.time()-start_onnx_time)
            
            # 后处理
            outputs = np.squeeze(outputs[0])
            boxes, scores, classes = self.postprocess(outputs, 
                                                      slice_info,
                                                      ratio,dw,dh)
            # print('单张 总用时',time.time()-start_time)
            record_time_slice.append(time.time()-start_time)
            
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_classes.extend(classes)
        
        # NMS处理
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.confidence, self.iou)
        
        # 绘制结果
        result_img = input_img.copy()
        for i in indices:
            box = all_boxes[i]
            score = all_scores[i]
            cls_id = all_classes[i]
            self.draw_box(result_img, box, score, cls_id)
        
        # print('总用时',time.time()-start_loop_time)
        record_time_all.append( time.time()-start_loop_time )
        
        return result_img,[record_time_slice_onnx,
                           record_time_slice,
                           record_time_all]

    def postprocess(self, outputs, slice_info,ratio,dw,dh):
        """后处理函数"""
        boxes, scores, classes = [], [], []
        
        rows = outputs.shape[0]
        
        for i in range(rows):
            conf = outputs[i][4]
            if conf < self.confidence: continue
            
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
 
            # 将框调整到原始图像尺寸，考虑缩放和填充
            x -= dw  # 移除填充
            y -= dh
            x /= ratio[0]  # 缩放回原图
            y /= ratio[1]
            w /= ratio[0]
            h /= ratio[1]
            left = int(x - w / 2)
            top = int(y - h / 2)
            # width = int(w)
            # height = int(h)
            # boxes.append([left, top, width, height])
            
            # 模型坐标 → 切片坐标
            # x = outputs[i][0] * target_w + self.pad_left
            # y = outputs[i][1] * target_h + self.pad_top
            # w = outputs[i][2] * target_w
            # h = outputs[i][3] * target_h
            
            # 切片坐标 → 原图坐标
            orig_x = slice_info['x_start'] + left
            orig_y = slice_info['y_start'] + top
            
            boxes.append([orig_x, orig_y, int(w), int(h)])
            scores.append(conf)
            classes.append(int(   np.argmax(outputs[i][5:])   ))
        
        return boxes, scores, classes

    def draw_box(self, img, box, score, cls_id):
        """绘制检测框"""
        x1, y1, w, h = box
        color = self.color_palette[cls_id]
        
        # 绘制边框
        cv2.rectangle(img,  (int(x1), int(y1)), 
                            (int(x1+w), int(y1+h)), 
                            color, 2)
        
        # 绘制标签
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(img, (label_x, label_y - label_height), 
                            (label_x + label_width, label_y + label_height), 
                            color, cv2.FILLED)
        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
 

if __name__ == "__main__":
    # for debug
    model = '/root/lsj/yolo11/runs/detect/train144/weights/best.onnx'
    input_img = "/root/datasets/mvTP-SIMO-Light-Aug/images/train/1_Down1_1_1_Right10-01-05rotate180.jpg"
    input_dir = "/root/datasets/mvYOLO-Det-TP-Aug/validate_whole/images/"
    slice_h = 2048
    slice_w = 2048
    conf_thres = 0.2
    iou_thres = 0.5
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=model)
    parser.add_argument("--input_img", default=input_img)
    parser.add_argument("--input_dir", default=input_dir)
    parser.add_argument("--slice_h", default=slice_h, type=int)
    parser.add_argument("--slice_w", default=slice_w, type=int)
    parser.add_argument("--conf_thres", default=conf_thres, type=float)
    parser.add_argument("--iou_thres", default=iou_thres, type=float)
    args = parser.parse_args()
    model = args.model
    input_img = args.input_img
    input_dir = args.input_dir
    slice_h = args.slice_h
    slice_w = args.slice_w
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    
    
    detector = YOLO11(model,)
    
    import os,shutil
    from os.path import join as osj
    from tqdm import tqdm
    if not args.input_dir or not os.path.exists(args.input_dir): # 不存在 图片文件夹，则按照单张图片处理
        result,_ = detector.detect(args.input_img,
                                args.slice_h,args.slice_w,
                                args.conf_thres,args.iou_thres,
                                )
        cv2.imwrite("output.jpg", result)
        print("检测完成，结果保存为 output.jpg")
    else:
        save_dir = f'./detect_res_{slice_h}'

        if os.path.exists(save_dir):shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        
        ''' time record '''
        record_time_slice_onnx=[]
        record_time_slice     =[]
        record_time_all       =[]
        for name in tqdm(os.listdir(args.input_dir)):
            if os.path.splitext(name)[-1] not in ('.jpg','.png','.bmp'):continue
            img_path = osj(args.input_dir,name)
            save_path = osj(save_dir,name)
            result,record_time_list = detector.detect(img_path,
                                args.slice_h,args.slice_w,
                                args.conf_thres,args.iou_thres,
                                )
            cv2.imwrite(save_path, result)
            
            record_time_slice_onnx.extend(record_time_list[0])
            record_time_slice.extend(record_time_list[1])
            record_time_all.extend(record_time_list[2])
            # print("检测完成，结果保存为 output.jpg")
            
        
        # plot_record_time
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,8))
        # record_time_slice_onnx = list(np.load('./record_time_slice.npy'))
        # record_time_slice = list(np.load('./record_single_time.npy'))
        # record_time_all = list(np.load('./record_time_all.npy'))
        for record_name,record in {
            'record_time_slice_onnx':record_time_slice_onnx,
            'record_time_slice':record_time_slice,
            'record_time_all':record_time_all,
        }.items():
            x_data = [i+1 for i in range(len(record))]
            plt.plot( x_data,record ,marker='o', linestyle='-',
                     label=record_name)
            
            plt.title(record_name)
            plt.xlabel('X idx')
            plt.ylabel('Y /s')
            
            plt.legend()
            plt.grid(True)
            plt.savefig( osj(save_dir,f'{record_name}.jpg') )
            plt.clf()
            
        
