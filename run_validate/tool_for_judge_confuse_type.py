''' 2025-3-20
主要任务 ：
1 漏检图片                           （红色）标注为 [cls_id]->OK
2 误检细分      
    2.1 没有缺陷 但是 检测出缺陷      （黄色） 标注为 OK->[cls_id]
    2.2 有缺陷 但是 检测类别出错      （蓝色） 标注为 [cls_id]->[cls_id]

保存漏检时  保存原图+标注    保存 圈画漏检区域（把原有的label标上去）
保存误检时  保存原图+标注    保存 圈画误检区域（把原有label-误检label 标上去）
'''
import os,cv2,shutil,argparse
from os.path import exists as is_path_exists
from os.path import join as osj
from collections import defaultdict
from tqdm import tqdm
# for debug
predict_root='runs/detect/predict6'
data_root="/root/datasets/mvYOLO-Det-TP-slice-Aug/validate_slice"
validate_img_dir = f'{data_root}/images' 
validate_label_dir = f'{data_root}/labels' 
pred_label_dir = f'{predict_root}/labels'
conf_threshold = 0.5

############## Start 参数传入 ####################
# 添加命令行参数
parser = argparse.ArgumentParser(description="检查模型预测结果的脚本——获得 漏检、误检 具体的图片标注")
parser.add_argument('--data_root',          type=str, default=data_root,help='验证文件根目录')
parser.add_argument('--validate_img_dir',     type=str, default=validate_img_dir,help='验证文件【图片】目录')
parser.add_argument('--validate_label_dir',    type=str, default=validate_label_dir,help='验证文件【label】目录，目录下txt文件')
parser.add_argument('--pred_label_dir',    type=str, default=pred_label_dir,help='预测label，目录下txt文件')
parser.add_argument('--conf_threshold',            type=float, default=conf_threshold, help='置信度')

# 解析命令行参数
args = parser.parse_args()

print(f'验证文件【图片】目录: {args.validate_img_dir}')
print(f'验证文件【label】目录，目录下txt文件: {args.validate_label_dir}')
print(f'预测label，目录下txt文件: {args.pred_label_dir}')
print(f'judge后文件存放位置: {args.data_root}')

data_root           = args.data_root
validate_img_dir    = args.validate_img_dir
validate_label_dir  = args.validate_label_dir
pred_label_dir      = args.pred_label_dir
conf_threshold      = args.conf_threshold
############## End 参数传入 ####################

cls2ok_color = (255,0,0)
ok2cls_color=(233, 186, 101)
cls2cls_color=(0,0,255)

cls2ok_dir=f'{data_root}/cls2ok' 
ok2cls_dir=f'{data_root}/ok2cls' 
cls2cls_dir=f'{data_root}/cls2cls' 
if is_path_exists(cls2ok_dir):shutil.rmtree(cls2ok_dir)
if is_path_exists(ok2cls_dir):shutil.rmtree(ok2cls_dir)
if is_path_exists(cls2cls_dir):shutil.rmtree(cls2cls_dir)
os.makedirs(cls2ok_dir)
os.makedirs(ok2cls_dir)
os.makedirs(cls2cls_dir)




'''
1 表示 缺陷 
0 表示 正常
'''
# 输出为 image_id, x y w h
def get_label_from_file(conf_threshold,filepath):
    if not is_path_exists(filepath): return (0,[],None)
    with open(filepath,mode='r',encoding='utf-8') as f:
        data = f.readlines()
    confs = [ float(d.split()[-1]) for d in data]
    max_conf = max(confs) if len(confs)!=0 else 0
    res_data = [ ' '.join(d.split()[:-1])  for d in data if float(d.split()[-1]) > conf_threshold ]
    return  (1 ,res_data,max_conf) if len(res_data) else (0,[],None)
# 输出为 image_id, x y w h
def get_label_from_gtfile(filepath):
    if not is_path_exists(filepath):return (0,[])
    with open(filepath,mode='r',encoding='utf-8') as f:
        data = f.readlines()
    return (1,data) if len(data) else (0,data)
# 读取label预测文件，格式假设为 image_id, x y w h
# 输出为  {cls_id : [[box1],[box2]...]} box=[xmin,ymin,xmax,ymax]
# def read_labels(file_path,height,width):
#     with open(file_path, 'r') as f:
#         data = f.readlines()
#     return process_labels(data,height,width)
# 格式假设为 image_id, x y w h
# 输出为  {cls_id : [[box1],[box2]...]} box=[xmin,ymin,xmax,ymax]
def process_labels(label_lines,height,width,): 
    labels = {}
    for line in label_lines:
        parts = line.strip().split()
        image_id = parts[0]
        box = list(map(float, parts[1:]))
        # 处理 坐标 逆归一化
        abs_x_center = box[0]*width
        abs_y_center = box[1]*height
        abs_width = box[2]*width
        abs_height = box[3]*height
        # 计算边界框坐标
        xmin = int(abs_x_center - abs_width / 2)
        ymin = int(abs_y_center - abs_height / 2)
        xmax = int(abs_x_center + abs_width / 2)
        ymax = int(abs_y_center + abs_height / 2)
        box = [xmin,ymin,xmax,ymax]
        if image_id not in labels:
            labels[image_id] = []
        labels[image_id].append(box)
    return labels

# 计算IoU
def calculate_iou(box1, box2):  # 需要  xmin ymin xmax ymax
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou
def check_whether_wrong_confuse_boxes(gt_labels,pred_labels,iou_threshold=0.5):
    # 遍历gt 寻找 pred 中有无 iou > 0.5 的框
    # 如果 没有 则 记作  [gt cls]->OK x y w h
    # 如果 有 
    #     但是 cls 不一致 则 记作  [gt cls]->[pred cls] x y w h
    #     如果 一致 则 continue
    wrong_boxes = defaultdict(list)
    confuse_boxes = defaultdict(list)
    
    for image_id, gt_boxes in gt_labels.items():
        pred_boxes = pred_labels.get(image_id, [])

        for gt_box in gt_boxes: # 遍历每一个gt 如果在pred中有重叠，就ok
            is_find=False
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    is_find=True
                    break
            if not is_find: # 没检查到这个box [cls]->OK
                wrong_boxes[f'{image_id}->OK'].append(gt_box)
        
        for gt_box in gt_boxes:
            for pred_id,pred_boxes in pred_labels.items():
                if pred_id == image_id:continue
                # 检查 在 其他类里面是否有 iou > 0.5 的 [cls]->[cls]
                for pred_box in pred_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou >= iou_threshold:
                        confuse_boxes[f'{image_id}->{pred_id}'].append(gt_box)
                        break
    return wrong_boxes,confuse_boxes
def draw_defect_annotation_ignore(img,boxes,color):
    # '[cls_id]->OK'
    for cls,bboxes in boxes.items():
        for (x1, y1, x2, y2) in bboxes:
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color , 2)
            # 添加类别标签
            label_text = f"{int(cls)}->OK"
            cv2.putText(img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        color, 2)
    return img
def draw_defect_annotation_overkill(img,boxes,color):
    # 'OK->[cls_id]'
    for cls,bboxes in boxes.items():
        for (x1, y1, x2, y2) in bboxes:
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color , 2)
            # 添加类别标签
            label_text = f"OK->{int(cls)}"
            cv2.putText(img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        color, 2)
    return img
def draw_defect_annotation_wrong_boxes(img,wrong_boxes,color):
    # '[cls_id]->OK'
    for label_text,boxes in wrong_boxes.items():
        for x1, y1, x2, y2 in boxes:
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color , 2)
            cv2.putText(img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        color, 2)
    return img
def draw_defect_annotation_confuse_boxes(img,confuse_boxes,color):
    # '[cls_id]->[cls_id]'
    for label_text,boxes in confuse_boxes.items():
        for x1, y1, x2, y2 in boxes:
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color , 2)
            cv2.putText(img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        color, 2)
    return img
        
# 最后保存 【图片名称】 在每个目录下
cls2ok_record=[]
cls2cls_record=[]
ok2cls_record=[]
for name in tqdm(os.listdir(validate_img_dir)):
    n,_ = os.path.splitext(name)
    img_path = osj(validate_img_dir,name)
    ori_img = cv2.imread(img_path)
    h,w,_ = ori_img.shape
    label_path = osj(validate_label_dir,n+'.txt')
    pred_label_path = osj(pred_label_dir,n+'.txt')
    
    # pred_res = 1 表示 缺陷 0 表示 正常
    pred_res,pred_data,max_conf = get_label_from_file(conf_threshold,pred_label_path)
    # gt_res = 1 表示 缺陷  0 表示 正常
    gt_res,gt_data = get_label_from_gtfile(label_path)
    pred_data = process_labels(pred_data,h,w) # {cls_id : [[box1],[box2]...]} box=[xmin,ymin,xmax,ymax]
    gt_data = process_labels(gt_data,h,w) # {cls_id : [[box1],[box2]...]} box=[xmin,ymin,xmax,ymax]
        
    if gt_res==1:
        if pred_res==0:
            # 有label 但是 没检测出来   漏检   [cls_id]->OK
            annotated_img = draw_defect_annotation_ignore(ori_img.copy(),gt_data,cls2ok_color)
            # 保存原图
            # 保存标注图
            cv2.imwrite(osj(cls2ok_dir,name),annotated_img)
            
            cls2ok_record.append(name+'\n')
        else:  # 有label 并且  检测出来 
            # wrong   [cls]->OK : [box1,box2...]  
            # confuse [cls]->[cls]:[box1,box2...]
            wrong_boxes_data,confuse_boxes_data = check_whether_wrong_confuse_boxes(gt_data,pred_data)
            if len(wrong_boxes_data):
                # wrong   [cls]->OK
                annotated_img = draw_defect_annotation_wrong_boxes(ori_img.copy(),wrong_boxes_data,
                                                                cls2ok_color)
                cv2.imwrite(osj(cls2ok_dir,name),annotated_img)
                cls2ok_record.append(name+'\n')
            if len(confuse_boxes_data):
                # confuse [cls]->[cls]
                annotated_img = draw_defect_annotation_confuse_boxes(ori_img.copy(),confuse_boxes_data,
                                                                    cls2cls_color)
                cv2.imwrite(osj(cls2cls_dir,name),annotated_img)
                cls2cls_record.append(name+'\n')
            # 保存原图
            # 保存标注图
    else:  # gt_res==0
        if pred_res==0:
            # 没有label 并且 没有检测出来  正确
            annotated_img = None
            pass
        else:
            # 没有label 但是 检测出来   误检
            # OK->[cls_id]
            annotated_img = draw_defect_annotation_overkill(ori_img.copy(),pred_data,ok2cls_color)
            # 保存原图
            # 保存标注图
            cv2.imwrite(osj(ok2cls_dir,name),annotated_img)
            ok2cls_record.append(name+'\n')
with open(osj(cls2ok_dir,'record.txt'),'w') as f:
    f.writelines(cls2ok_record)
with open(osj(cls2cls_dir,'record.txt'),'w') as f:
    f.writelines(cls2cls_record)
with open(osj(ok2cls_dir,'record.txt'),'w') as f:
    f.writelines(ok2cls_record)
    
res_msg = \
f'''
ignore      {len(cls2ok_record)}
confuse     {len(cls2cls_record)}
overkill    {len(ok2cls_record)}
'''
print(res_msg)
