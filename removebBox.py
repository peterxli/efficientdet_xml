#coding=utf-8
import cv2
import string
import random
import os
import numpy as np
import struct
import shutil
from tqdm import tqdm
import math
import PIL.Image
import copy



def remove_bbox_V2(folder_dir, pixel_threshold=4):
    """
    先找到像素值在对应区间的像素,然后bfs搜索当前像素块单方向是否满足长度超过阈值,将满足的像素块全部保存起来,然后遍历修改原图中对应像素块所在位置的像素的颜色
    :param folder_dir:
    :param pixel_threshold:
    :return:
    """
    # save_folder_dir = "./bbox_handle"

    color_threshold = [140, 100, 90]  # BGR
    sub_folder_name_list = os.listdir(folder_dir)
    # for sub_folder_name in sub_folder_name_list:
    for sub_folder_name in tqdm(sub_folder_name_list, total=len(sub_folder_name_list), ncols=80, leave=False):
        sub_folder_dir = os.path.join(folder_dir, sub_folder_name)
        file_name_list = os.listdir(sub_folder_dir)
        for file_name in file_name_list:
        # for file_name in tqdm(file_name_list, total=len(file_name_list), ncols=80, leave=False):
            # if "745.jpg" in file_name:

            if ".jpg" in file_name:

                file_dir = os.path.join(sub_folder_dir, file_name)
                save_file_name = file_name.split('.')[0]+"_mvbox.jpg"
                save_dir = os.path.join(sub_folder_dir, save_file_name)
                # print(file_dir)
                # print(save_dir)
                # quit()
                try:
                    img = cv2.imread(file_dir)
                except Exception as e:
                    print(e)
                    continue
                if len(np.shape(img)) != 3:
                    print(file_dir)
                    continue
                img_h, img_w, img_c = np.shape(img)
                candidate_bbox_image = np.zeros((img_h, img_w))
                final_bbox_image = np.zeros((img_h, img_w))
                path_label = [[False for _ in range(img_w)] for _ in range(img_h)]
                handle_img = copy.deepcopy(img)

                #  找到所有颜色区间满足的像素点保存到candidate_bbox_image  同时生成path_label
                for h in range(img_h):
                    for w in range(img_w):
                        pixel = img[h][w]
                        if pixel[0] < color_threshold[0] and pixel[1] > color_threshold[1] and pixel[2] < color_threshold[2]:
                            candidate_bbox_image[h][w] = 255
                            path_label[h][w] = True

                search_length = 4

                for h in range(img_h):
                    for w in range(img_w):
                        if path_label[h][w]:

                            #  水平查找
                            queue = []
                            queue.append([h, w])
                            block_pixel_sum = 1

                            dires = [[0, 1]]
                            APPEND_FLAG = False
                            while queue:
                                y, x = queue.pop(0)
                                for dy, dx in dires:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < img_w and 0 <= ny < img_h and path_label[ny][nx]:
                                        block_pixel_sum += 1
                                        queue.append([ny, nx])
                                        if block_pixel_sum > pixel_threshold:
                                            APPEND_FLAG = True
                                            break

                            queue = []
                            queue.append([h, w])
                            while queue:
                                y, x = queue.pop(0)

                                if APPEND_FLAG:
                                    final_bbox_image[y, x] = 255

                                for dy, dx in dires:
                                    nx, ny = x + dx, y + dy
                                    # path_label[y][x] = False

                                    for i in range(search_length*2 + 1):
                                        i -= search_length
                                        if 0 <= y + i < img_h and candidate_bbox_image[y + i][x] != 255:
                                            handle_img[y][x] = handle_img[y + i][x]
                                            break

                                    if 0 <= nx < img_w and 0 <= ny < img_h and path_label[ny][nx]:
                                        queue.append([ny, nx])

                            #  垂直查找
                            queue = []
                            queue.append([h, w])
                            block_pixel_sum = 1

                            dires = [[1, 0]]
                            APPEND_FLAG = False
                            while queue:
                                y, x = queue.pop(0)
                                for dy, dx in dires:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < img_w and 0 <= ny < img_h and path_label[ny][nx]:
                                        block_pixel_sum += 1
                                        queue.append([ny, nx])
                                        if block_pixel_sum > pixel_threshold:
                                            APPEND_FLAG = True
                                            break

                            queue = []
                            queue.append([h, w])
                            while queue:
                                y, x = queue.pop(0)

                                if APPEND_FLAG:
                                    final_bbox_image[y, x] = 255

                                for dy, dx in dires:
                                    nx, ny = x + dx, y + dy
                                    # path_label[y][x] = False

                                    for i in range(search_length*2 + 1):
                                        i -= search_length
                                        if 0 <= x + i < img_w and candidate_bbox_image[y][x + i] != 255:
                                            handle_img[y][x] = handle_img[y][x + i]
                                            break

                                    if 0 <= nx < img_w and 0 <= ny < img_h and path_label[ny][nx]:
                                        queue.append([ny, nx])


                # cv2.imwrite("./candidate_bbox_image.jpg", candidate_bbox_image)
                # cv2.imwrite("./final_bbox_image.jpg", final_bbox_image)
                # cv2.imwrite("./handle_img.jpg", handle_img)
                # quit()


                cv2.imwrite(save_dir, handle_img)
                # quit()


if __name__ == '__main__':
    remove_bbox_V2("/home/hualai/ponder/16_Magik/Models/pytorch/persondet/data/sample", 4)