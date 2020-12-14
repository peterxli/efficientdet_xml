# -*- coding: utf-8 -*-
import cv2
import os


videos_src_path = 'C:\\Users\\Zhang\\Desktop\\aaa\\buy\\'  # 提取图片的视频文件夹
dirs = os.listdir(videos_src_path)  # 获取指定路径下的文件
count = 0

# 循环读取路径下的文件并操作
for video_name in dirs:
    outputPath = "C:\\Users\\Zhang\\Desktop\\aaa\\buy_img\\"
    print("start\n")
    print(videos_src_path + video_name)
    vc = cv2.VideoCapture(videos_src_path + video_name)
    ret, frame = vc.read()
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    # 每隔n帧保存一张图片
    frame_interval = 4
    # 统计当前帧
    frame_count = 1
    while ret:
        ret, frame = vc.read()
        if frame_count % frame_interval == 0:
            if frame is not None:
                filename = outputPath + "{:0>12}".format(count)
                cv2.imwrite(filename, frame)
                count += 1
                print("保存图片:{}".format(filename))
        frame_count += 1
    vc.release()
    print("[INFO] 总共保存：{}张图片\n".format(count))
