# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :detect_object
# @File     :flur_detect.py
# @Date     :2020/11/9 2:46 下午
# @Author   :peterxli
-------------------------------------------------
"""
import cv2
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


# 模糊视频检测算法
def de_f(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    gray = cv2.convertScaleAbs(gray)
    mv, sd = cv2.meanStdDev(gray)
    sd = float(sd)
    return sd * sd


def detect_video(video_path):
    av_list = []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        if frame is not None:
            frame_num = de_f(frame)
            av_list.append(frame_num)
            average_num = np.mean(av_list)
            return average_num


if __name__ == '__main__':
    ori_videoFile = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/0_video/first_2000_video'
    save_videoFile = '/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/res/'
    file_list = os.listdir(ori_videoFile)
    file_list.sort()
    flur_list = []
    y_list = []
    plt.title('demo')
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    for file in file_list:
        file_path = os.path.join(ori_videoFile, file)
        print(file_path)
        num = detect_video(file_path)
        if num < 1900:
            shutil.copy(file_path, save_videoFile)
        if num is None:
            pass
        else:
            flur_list.append(num)
        if len(flur_list) > 10000:
            flur_list_average = np.mean(flur_list)
            flur_list_std = np.std(flur_list, ddof=1)
            for num in flur_list:
                y = normfun(num, flur_list_average, flur_list_std)
                y_list.append(y)
            print('average', flur_list_average)
            print('std', flur_list_std)
            plt.scatter(flur_list, y_list)
            plt.savefig('test.jpg')
            quit()

