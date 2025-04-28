import os,cv2
from os.path import join as osj
import shutil
from tqdm import tqdm  
import argparse 

parser = argparse.ArgumentParser(description="验证过杀漏检的脚本")

'''
图片角度检测逻辑（筛选conf大于conf_threshold的pred）
    遍历gt与pred，如果 gt 有 label return 1 else 0
                 如果 pred 有 label return 1 else 0
    if gt==pred: ok_num+=1
    elif gt == 1 and pred == 0: 漏检+=1
    elif gt == 0 and pred == 1: 过杀+=1
return ok_num, 漏检数量 ， 过杀数量
    
box角度检测逻辑（筛选conf大于conf_threshold的pred）
    遍历gt的box，记作 boxA，
        只要pred boxes存在一个boxB，使得calcIOU(boxA,boxB) > 0.5
            ok_box += 1
return  预测成功的box数量 / gt的box总数量
'''
# for debug
predict_root='runs/detect/predict6'
data_root="/root/datasets/mvYOLO-Det-TP-slice-Aug/validate_slice"
predict_dir=f'{predict_root}/labels'
target_label_dir=f'{data_root}/labels' 
ori_img_dir=f'{data_root}/images' 
save_img_dir=f'{data_root}/val1' 
conf=0.8

# 添加命令行参数
parser.add_argument('--predict_dir',     type=str, default=predict_dir, help='预测好的文件夹，文件为txt')
parser.add_argument('--target_label_dir',type=str, default=target_label_dir,help='目标label文件夹，文件为txt')
parser.add_argument('--ori_img_dir',     type=str, default=ori_img_dir,help='原来的图片文件夹')
parser.add_argument('--save_img_dir',    type=str, default=save_img_dir,help='overkill或ignore后，文件存放的文件夹')
parser.add_argument('--conf',            type=float, default=conf, help='置信度')

# 解析命令行参数
args = parser.parse_args()

print(f'预测好的文件夹，文件为txt: {args.predict_dir}')
print(f'目标label文件夹，文件为txt: {args.target_label_dir}')
print(f'原来的图片文件夹: {args.ori_img_dir}')
print(f'overkill或ignore后，文件存放的文件夹: {args.save_img_dir}')

predict_dir         = args.predict_dir
target_label_dir        = args.target_label_dir
ori_img_dir         = args.ori_img_dir
save_img_dir        = args.save_img_dir
conf        = args.conf

if os.path.exists(save_img_dir):shutil.rmtree(save_img_dir)
os.makedirs(save_img_dir)

'''
det 漏检 过杀 逻辑
图片没有instance 但是检测出来的    过杀 overkill
图片有instance   但是没有检测出来  漏检 ignore
'''
# 检查是否是 overkill 或 ignore 的函数
def check_image_status(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return 'overkill'
    elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
        return 'ignore'
    else:
        return 'ok'  # 没有overkill或ignore
# 匹配过程
def check_overkill_or_ignore(gt_file, pred_file):
    gt_labels = read_labels(gt_file)
    pred_labels = read_labels(pred_file)

    results = {}
    
    for image_id in gt_labels.keys():
        gt_boxes = gt_labels.get(image_id, [])
        pred_boxes = pred_labels.get(image_id, [])

        result = check_image_status(gt_boxes, pred_boxes)
        results[image_id] = result  # 图 

    return results


ok_num,overkill,ignore = 0,0,0

# 1 predict中的预测结果 == label ？
# 2 if == 则 计数
#   else 根据 label 选择 overkill文件夹/ignore文件夹 并且 计数
# 3 存放 不相等 的图片 到 对应文件夹中

def get_label_from_file(conf_threshold,filepath):
    with open(filepath,mode='r',encoding='utf-8') as f:
        data = f.readlines()
    confs = [ float(d.split()[-1]) for d in data]
    max_conf = max(confs) if len(confs)!=0 else 0
    res_data = [ ' '.join(d.split()[:-1])  for d in data if float(d.split()[-1]) > conf_threshold ]
    return  (1 ,res_data,max_conf) if len(res_data) else (0,[],None)

def get_label_from_gtfile(filepath):
    if not os.path.exists(filepath):return (0,[])
    with open(filepath,mode='r',encoding='utf-8') as f:
        data = f.readlines()
    return (1,data) if len(data) else (0,data)

"""

"""
names = os.listdir(predict_dir)
def save(mode,img_name):
    final_save_img_dir = osj(save_img_dir,mode)
    os.makedirs(final_save_img_dir,exist_ok=True)
    ori_img_path = osj(ori_img_dir,img_name)
    dst_img_path = osj(final_save_img_dir,img_name)
    shutil.copy2(ori_img_path,dst_img_path)

for name in tqdm(names):
    n,ext = name.split('.')
    img_name = n+'.jpg'
    label_path = osj(predict_dir,name)
    gt_label_path = osj(target_label_dir,name)
    # 获取 gt 内容 存在box 则 1 否则 0
    # 获取 pred 内容 存在box 则 1 否则 0 （conf筛选）
    # gt_res == pred == 1 预测成功
    # gt_res == pred == 0 预测成功
    # gt_res =1  pre ==0 漏检
    # gr_res =0  pred=1 过杀
    pred_res,_,max_conf = get_label_from_file(conf,label_path)
    gt_res,_ = get_label_from_gtfile(gt_label_path)

    if gt_res==pred_res:
        ok_num+=1
    elif gt_res==1:
        ignore+=1
        print('ignore 时的conf max',max_conf)
        save('ignore',img_name)
    elif gt_res==0:
        save('overkill',img_name)
        print('overkill 时的conf max',max_conf)
        overkill+=1
    else:pass

print('*'*10,'图片角度的过杀漏检：','*'*10)
print(f'ok:{ok_num}')
print(f'overkill:{overkill}')
print(f'ignore:{ignore}')
print('*'*20)
#####################################################################
#####################################################################
#####################################################################

# 计算IoU
def calculate_iou(box1, box2):
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

# 读取label文件和预测文件，格式假设为 image_id, x y w h
def read_labels(file_path,height,width):
    if not os.path.exists(file_path):return {}
    with open(file_path, 'r') as f:
        data = f.readlines()
    return process_labels(data,height,width)
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

# 匹配过程
def match_boxes(gt_file, pred_file,height,width, iou_threshold=0.5,):
    gt_labels       = read_labels(gt_file,height,width,) # gt dict  包含  每个id  对应的  box
    _,pred_labels,_ = get_label_from_file(conf,pred_file) # 这一步做conf筛选
    pred_labels     = process_labels(pred_labels,height,width)       # pred dict  包含  每个id  对应的  box

    total_gt_boxes = 0
    matched_boxes = 0

    for image_id, gt_boxes in gt_labels.items():
        pred_boxes = pred_labels.get(image_id, [])
        total_gt_boxes += len(gt_boxes)

        for gt_box in gt_boxes: # 遍历每一个gt 如果在pred中有重叠，就ok
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    matched_boxes += 1
                    break

    return matched_boxes,total_gt_boxes

# 输出结果
names = os.listdir(predict_dir)
all_matched_boxes = 0
all_total_gt_boxes = 0
for name in tqdm(names):
    n,ext = os.path.splitext(name)
    img_name = n+'.jpg'
    img_path = osj(ori_img_dir,img_name)
    h,w,_ = cv2.imread(img_path).shape
    pred_file = osj(predict_dir,name)
    gt_file = osj(target_label_dir,name)
    matched_boxes,total_gt_boxes = match_boxes(gt_file, pred_file,h,w)
    # 统计
    all_matched_boxes+=matched_boxes
    all_total_gt_boxes += total_gt_boxes
box_match_ratio = all_matched_boxes / all_total_gt_boxes if all_total_gt_boxes > 0 else 0
print('*'*10,'instance box角度的过杀漏检：','*'*10)
print(f"匹配的box数量: {all_matched_boxes}")
print(f"总的的box数量: {all_total_gt_boxes}")
print(f"匹配的box占总数比例: {box_match_ratio * 100:.2f}%")
print('*'*20)
