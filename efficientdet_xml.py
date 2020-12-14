# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Date     :2020/8/31 10:20 上午
# @Author   :peterxli
-------------------------------------------------
"""
import torch
from torch.backends import cudnn
import os
from backbone import EfficientDetBackbone
import numpy as np
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
from pathlib import Path
import xml.etree.cElementTree as ET
from PIL import Image
import shutil
from rename_img import rename_photo
import del_green

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
compound_coef = 6

img_path = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/1203photo'
save_img_path = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/all_image'
save_path = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/1203xml/'
number = 0

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()


def automate(file):
    tempshapes = []
    flag = False
    threshold = 0.2
    iou_threshold = 0.2
    # color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef]
    ori_imgs, framed_imgs, framed_metas = preprocess(file, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue

        ori_imgs[i] = ori_imgs[i].copy()

        for j in range(len(out[i]['rois'])):
            x1, y1, x2, y2 = out[i]['rois'][j].astype(np.int)
            x1 = str(x1)
            x2 = str(x2)
            y1 = str(y1)
            y2 = str(y2)
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])
            if score < 0.3:
                continue
            if obj in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'cat', 'dog']:
                flag = True
                if obj in ['car', 'bus', 'truck']:
                    obj = 'car'
                elif obj in ['bicycle', 'motorcycle']:
                    obj = 'bicycle'
                else:
                    obj = obj
                points = x1, y1, x2, y1, x2, y2, x1, y2, obj
                str_flag = ','
                a = str_flag.join(points)
                tempshapes.append(a)
            else:
                break
            # print(tempshapes)
    return tempshapes, flag


def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
         if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def create_labimg_xml(image_path, annotation_list):
    image_path = Path(image_path)
    new_save_path = Path(save_img_path)
    img = np.array(Image.open(image_path).convert('RGB'))
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(new_save_path.name)
    # ET.SubElement(annotation, 'folder').text = str(image_path.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path.name)
    ET.SubElement(annotation, 'path').text = str(os.path.join(new_save_path, image_path.name))
    # ET.SubElement(annotation, 'path').text = str(image_path)
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img.shape[1])
    ET.SubElement(size, 'height').text = str(img.shape[0])
    ET.SubElement(size, 'depth').text = str(img.shape[2])

    ET.SubElement(annotation, 'segmented').text = '0'
    if annotation_list is None:
        return
    else:
        for annot in annotation_list:
            tmp_annot = annot.split(',')
            cords, label = tmp_annot[0:-2], tmp_annot[-1]
            xmin, ymin, xmax, ymax = cords[0], cords[1], cords[4], cords[5]

            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = label
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'

            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
            indent(annotation)
        tree = ET.ElementTree(annotation)
        xml_file_name = save_path + (image_path.name.split('.')[0] + '.xml')
        tree.write(xml_file_name)


def main():
    img_filename = os.listdir(img_path)
    img_filename.sort()
    for i in range(number, len(img_filename)):
        ori_filename = os.path.join(img_path, img_filename[i])
        Flag = del_green.re_green_photo(ori_filename)
        if Flag:
            vetor, flag = automate(ori_filename)
            if flag:
                new_name = rename_photo()
                new_filename = os.path.join(img_path, new_name+'.jpg')
                print('new_filename', new_filename)
                os.rename(ori_filename, new_filename)
                create_labimg_xml(new_filename, vetor)
                shutil.move(new_filename, save_img_path)
                print(new_filename)
            else:
                os.remove(ori_filename)
        else:
            continue


if __name__ == '__main__':
    main()

