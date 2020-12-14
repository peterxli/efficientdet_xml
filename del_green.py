#coding=utf-8
import cv2
import os
import numpy as np
import shutil
from PIL import Image


def remove_bbox(folder_dir):
    # save_folder_dir = "./bbox_handle"

    color_threshold = [140, 210, 90]  # BGR
    img_name_list = os.listdir(folder_dir)
    img_name_list.sort()
    # for sub_folder_name in sub_folder_name_list:

    for img_name in img_name_list:
        file_name = os.path.join(folder_dir, img_name)
        if ".jpg" in img_name:
            save_file_name = img_name.split('.')[0]+".jpg"

            save_dir = os.path.join(folder_dir + '/save', save_file_name)
            try:
                img = cv2.imread(file_name)
                img = cv2.resize(img, (320, 240))
            except Exception as e:
                print(e)
                continue
            if len(np.shape(img)) != 3:
                print(file_dir)
                continue
            img_h, img_w, img_c = np.shape(img)
            candidate_bbox_list = []

            #  找到所有颜色区间满足的像素点保存到candidate_bbox_image  同时生成path_label

            for h in range(img_h):
                for w in range(img_w):
                    pixel = img[h][w]
                    if pixel[0] < color_threshold[0] and pixel[1] > color_threshold[1] and pixel[2] < color_threshold[2]:
                        candidate_bbox_list.append((h, w))
            if len(candidate_bbox_list) >= 25:
                print('the delete photo name is', img_name)
                shutil.move(file_name, save_dir)


# 检测带绿框图片并删除，对无绿框图片进行resize
def re_green_photo(file_name):
    flag = True
    color_threshold = [140, 200, 90]  # BGR
    img = cv2.imread(file_name)
    crop_img = img[0:972, 0:1920]
    img = cv2.resize(crop_img, (320, 240))
    if len(np.shape(img)) != 3:
        pass
    img_h, img_w, img_c = np.shape(img)
    candidate_bbox_list = []
    for h in range(img_h):
        for w in range(img_w):
            pixel = img[h][w]
            if pixel[0] < color_threshold[0] and pixel[1] > color_threshold[1] and pixel[2] < color_threshold[2]:
                candidate_bbox_list.append((h, w))
    if len(candidate_bbox_list) >= 15:
        flag = False
        print('the delete photo name is', file_name)
        os.remove(file_name)
    else:
        cv2.imwrite(file_name, crop_img)
    return flag



