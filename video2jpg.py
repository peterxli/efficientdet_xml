'''
实现目的：将MP4视频文件抽帧为图片
    1.步骤一：cv2读取视频,2秒（40帧左右）抽取一张,按照文件数目命名
    2.步骤二：使用Resnet模型，滤除背景图片，仅留下有检测结果的照片
    3.步骤三：对剩余照片，重新按照数目进行标号。

    新增功能，向原本的视频文件夹中添加新的视频，如何不破坏原有图片文件夹
    4.

    疑问：
    1.为什么不能直接预测frame呢？因为frame不是图片，导致模型运行不通，所以还是先全部保存图片再检测，再删除
    2.

    重要特性：
    1.加入新视频，运行程序，图片文件夹会保留已经处理过的图片，在基础上继续增加图片
    2.同时，AI模型预测时，过滤掉之前已经预测过的图片，只从最新照片开始预测
    3.


    可以继续优化的方向：Resnet模型预测了80类，但我们只需要5类，可以预测的时候只筛选相近的类别。如car,bus,coath,其余在labelimg中不显示


时间：2020.06.10
作者：xjy
'''

import cv2
import os
import shutil
from PIL import Image, ImageTk
import keras
import config
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image
import numpy as np
import tensorflow as tf


#开启tf的session
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#加载模型
keras.backend.tensorflow_backend.set_session(get_session())
model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')#当前路径下放个模型
model = models.load_model(model_path, backbone_name='resnet50')

def load_tfimage(file):
    img = Image.open(file)
    # Resize to Pascal VOC format
    w, h = img.size
    img_height = h
    img_width = w
    if w >= h:
        baseW = 500
        wpercent = (baseW / float(w))
        hsize = int((float(h) * float(wpercent)))
        img = img.resize((baseW, hsize), Image.BICUBIC)
        return img
    else:
        baseH = 500
        wpercent = (baseH / float(h))
        wsize = int((float(w) * float(wpercent)))
        img = img.resize((wsize, baseH), Image.BICUBIC)
        return img

#进行一次推演，得到框，类别，置信度
def automate(file):
    img = load_tfimage(file)
    tempshapes = []
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    opencvImage = open_cv_image[:, :, ::-1].copy()
    # opencvImage = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
    image = preprocess_image(opencvImage)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    for idx, (box, num_label, score) in enumerate(zip(boxes[0], labels[0], scores[0])):
        if score < 0.5:
            continue

        # if config.labels_to_names[label] not in curr_label_list:
        #   continue
        label = config.labels_to_names[num_label]
        w, h = img.size
        if w >= h:
            baseW = 500
            wpercent = (float(w) / baseW)
            box = box * wpercent
        else:
            baseH = 500
            wpercent = (float(h) / baseH)
            box = box * wpercent
        b = box.astype(int)
        difficult = False
        points = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
        xjy_normal_format_points = [(b[0], b[1]), (b[2], b[3])]
        tempshapes.append((label, xjy_normal_format_points, score, None, difficult))
        # tempshapes.append((label, points, None, None, difficult))
    # print(tempshapes)
    return(tempshapes)



    #
    #     # if config.labels_to_names[label] not in curr_label_list:
    #     #   continue
    #     label = config.labels_to_names[num_label]
    #     w, h = self.img.size
    #     if w >= h:
    #         baseW = 500
    #         wpercent = (float(self.img_width) / baseW)
    #         box = box * wpercent
    #     else:
    #         baseH = 500
    #         wpercent = (float(self.img_height) / baseH)
    #         box = box * wpercent
    #     b = box.astype(int)
    #     difficult = False
    #     points = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
    #     tempshapes.append((label, points, None, None, difficult))
    # print(tempshapes)
    # self.loadLabels(tempshapes)
    # self.setDirty()


#按照你的视频路径进行更改
# your_video_path = "/home/hualai/ponder/1_pycharm/02-project/10video_name_num/0_video"
# your_video_path = "/home/hualai/ponder/1-data/data_2000_video/2000_video"
# your_video_path = "/home/hualai/ponder/1-data/data_2000_video/2000_video_test_2"
#6月.23号第一次上线测试
your_video_path = "//192.168.107.58/share/hualai_image_label_project/0_video/10.19"


'''
模拟测试：
    1.2000个视频分两批，模拟两次任务，一次100张
    2.同时一次性跑一次2000,两次看对比结果。无差异，证明程序没问题
    
    3.实验结果证明无问题。1063+1148 = 2211

但又想到一个问题：2020.6.11下午1点
    1.这次2000视频有绿框，需要删除，但删除后，又打乱了排序。即使不带绿色框，万一也会出现一些奇怪的图片，需要删除怎么办？
    2.还是应该跑完这个video2jpg后人工检查一遍图片，之后再来一次排序。
'''

# your_jpg_path = '/home/hualai/ponder/1_pycharm/02-project/10video_name_num/1'
# your_jpg_path = '/home/hualai/ponder/1_pycharm/02-project/10video_name_num/hualai_label_0'
# your_jpg_path = '/home/hualai/ponder/1_pycharm/02-project/10video_name_num/compare_result/1000_2'
#6月.23号第一次上线测试
your_jpg_path = '//192.168.107.58/share/hualai_image_label_project/1019photo'

#定义主函数，防止其它程序调用时，运行本程序
# def main():
'''
标志位，区别是不是第一次运行程序
'''
first_or_not_button = 0
first_or_not = os.listdir(your_jpg_path)

#拿上次最大数


if len(first_or_not) > 0:
    print("非空图片目录，不是第一次运行")
    first_or_not_button = 1
    max_last_jpg = max(first_or_not).split(".")[0]
    print("fi-----------------------------max_last_jpg :", max_last_jpg)
else:
    print("空目录, 这是第一次运行程序")
    first_or_not_button = 0


# ！！！！！！！！！暂时清空图片目录------仅调试用，实际勿用!!!!!!!!!!!!!!
# shutil.rmtree(your_jpg_path)
# os.mkdir(your_jpg_path)

all_video = [0]  # 暂存视频名称列表

i = 0
all_video = os.listdir(your_video_path)#读取your_video_path目录下的所有视频名称
print("共%d个视频" % len(all_video))


#更新逻辑，读视频，写照片前，读一下jpg文件下的最大命名数

# max_num_hold = [[] for _ in range(0)]


#视频抽帧
def capFrame(videoPath, savePath):

    j = 0
    cap = cv2.VideoCapture(videoPath)

    if cap.isOpened():  # 判断是否正常打开
        print("这个视频正常打开")
        rval, frame = cap.read()
    else:
        rval = False
        print("!!!!!!!!!!!!!!!!这个视频打不开!!!!!!!!!!")


    numFrame = 0
    while rval:
        rval, frame = cap.read()
        numFrame += 1
        # 每10桢截取一个图片

        #太多了，所以40帧一张吧，华来的视频：20 帧/秒
        if numFrame % 100 == 1:
            new_num = len(os.listdir(your_jpg_path))#计数你所存图片的文件夹，以计数作为命名，但是存在一个问题，就是放新视频进来，可能会冲突。

            print("new_num    : ", new_num)
            new_num_12 = "{:0>12}".format(new_num)
            print("new_num_12 : ",new_num_12 )

            newPath = savePath + str(new_num_12) + ".jpg"
            print("write Path : ", newPath)
            j += 1
            all_jpg_num.append(j)

            '''
            判断一下，有图片，不写
            ！更新逻辑，加入新加入视频，导致新视频抽取的图片与原始冲突
            可以在原始图片最大数目上继续
            '''
            isExists = os.path.exists(newPath)
            print("isExists  :  ", isExists)

            isExists_count = 0
            #max_num_hold=(max(os.listdir(your_jpg_path)))
            #print("max_num_hold : ", max_num_hold)

            if isExists == True:
                print("图片冲突，跳过此图片 ： ",newPath)
                # continue
                print("更新逻辑，最大数目上继续标号")

                num = len(os.listdir(your_jpg_path))

                isExists_count += 1
                #从最后一张图片命名，接着开始命名，如最后一张0000013,则开始000014命名
                #不破坏已经处理过的数据
                new_num_12 = "{:0>12}".format(num + isExists_count)
                print("new_num_12_update    : ", new_num_12)
                newPath = savePath + str(new_num_12) + ".jpg"
                print("new_new_path : ", newPath)

                cv2.imwrite(newPath, frame)



            else:
                print("write Path : ", newPath)
                if frame is None:
                    print("!!!!!!!!error  the frame is none!!!!!!!!!!")
                    continue
                else:
                    cv2.imwrite(newPath, frame)


    print("这个视频共转换图片%d张:"% len(all_jpg_num))
    cv2.waitKey(1)
    cap.release()
    return all_jpg_num


def sum_list(items):
    sum_numbers = 0
    for x in items:
        sum_numbers += x
    return sum_numbers


b = []
for single_video in all_video:
    all_jpg_num = []
    i += 1
    print("第%d个视频" % i, "视频名称：", single_video)
    a = capFrame(your_video_path + "/" + single_video, your_jpg_path + "/" )
    b.append(len(a))
print("总共视频图像数目：", sum_list(b))


#第二步 ： 开始检测图像

'''
逻辑梳理：
    1.首先判断图片文件夹是否为空，如果空，则全部AI推演
    2.非空：拿最大图片号开始推演，避免重复推演之前已经推演过的图片
    3.放一个标志位first_or_not_button
        first_or_not_button =1 ：拿最大数
        first_or_not_button =0 ：全部参与推演
'''


delete_num = 0
file = os.listdir(your_jpg_path)
file.sort(key = lambda x:int(x[:-4]))

total_befor_delete = len(file)

our_label_list = ["person", "bicycle", "car", "motorbike", "truck", "bus", "cat", "dog"]

# print("file_xxxxxxxxxxxxxxxxxxxx : ",file)

if first_or_not_button == 0:
    print("首次运行程序，全部推演")
    for img in os.listdir(your_jpg_path):
        img_path = os.path.join(your_jpg_path, img)
        print("img_path : ", img_path)
        tempshapes=automate(img_path)
        print("tempshapes", tempshapes)

        #更新逻辑，只要人，车，狗，猫，非机动车
        if tempshapes:

            # pass
            print("这张图片有目标，但要筛选 :", img_path)
            print("tempshapes[0][0] : ", tempshapes[0][0])
            if tempshapes[0][0] in our_label_list:
                print("这张照片我们需要标注 ： ",img_path)
                pass

            else:
                print("这张图片有目标，但不关心，删除 :", img_path)
                print("tempshapes[0][0] : ", tempshapes[0][0])
                os.remove(img_path)
                delete_num += 1
                print("共删除 :", delete_num)
                print("剩余 :", total_befor_delete - delete_num)

        else:
            print("这张图片无目标，准备删除 :", img_path)
            os.remove(img_path)
            delete_num +=1
            print("共删除 :", delete_num)
            print("剩余 :", total_befor_delete -delete_num)

else:
    print("拿上次最大，推演")
    for img in os.listdir(your_jpg_path):
        # print("img_xxxxxxxxxxxxxxxxxx : ", img)
        # img = img.split(".")[0]
        # print("img_xxxxxxxxxxxxxxxxxx : ", img)

        img_number = img.split(".")[0]#拿到所有图片名号

        if int(img_number) < int(max_last_jpg):#图片名中小于上次最大号的，跳过，不用预测
            print("img：{} ,小于之前最大号的,  最大号是{}。所以这张图片不参与此次模型筛选。".format(img_number, max_last_jpg))
            continue
        else:

            img_path = os.path.join(your_jpg_path, img)
            print("img_path : ", img_path)

            tempshapes = automate(img_path)
            print("tempshapes", tempshapes)

            # 更新逻辑，只要人，车，狗，猫，非机动车
            if tempshapes:

                # pass
                print("这张图片有目标，但要筛选 :", img_path)
                print("tempshapes[0][0] : ", tempshapes[0][0])
                if tempshapes[0][0] in our_label_list:
                    print("这张照片我们需要标注 ： ", img_path)
                    pass

                else:
                    print("这张图片有目标，但不关心，删除 :", img_path)
                    print("tempshapes[0][0] : ", tempshapes[0][0])
                    os.remove(img_path)
                    delete_num += 1
                    print("共删除 :", delete_num)
                    print("剩余 :", total_befor_delete - delete_num)

            else:
                print("这张图片无目标，准备删除 :", img_path)
                os.remove(img_path)
                delete_num += 1
                print("共删除 :", delete_num)
                print("剩余 :", total_befor_delete - delete_num)



#第三步 ： 针对剩余的图片，按照图片数目重新排号
 # 计数你所存图片的文件夹，以计数作为命名，但是存在一个问题，就是放新视频进来，可能会冲突。


total = len(os.listdir(your_jpg_path))
print("total : ", total)
filelist = os.listdir(your_jpg_path)

'''
更新，解决filelist无序问题
'''
filelist.sort(key = lambda  x:int(x[:-4]))##倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大排序

count = 0
ii = 0

num_list = []
for img in filelist:

    print("img", img)
    #更新逻辑，拿到最大的图片命名
    #img = img.split(".")[0]
    #num_list.append(int(img))
    #print("num_list : ", num_list)
    #max_num = max(num_list)
    #print("max : ", max_num)

    # if count <= total:
    #
    #     newname = "{:0>12}".format(i)
    #     print("newname : ", newname)
    #     count += 1

    if ii <= total:
        newname = "{:0>12}".format(ii)
        print("newname : ", newname)
        ii += 1

        src = os.path.join(os.path.abspath(your_jpg_path),img)
        # print("src : ", src)
        dst = os.path.join(os.path.abspath(your_jpg_path),newname + '.jpg')
        # print("dst : " ,dst)
        os.rename(src, dst)

        #更新逻辑，如果图片数目跟标号一致，别rename
        # if len(os.listdir(your_jpg_path)) == int():
        #     pass
        # else:
        #     os.rename(src, dst)#遇见问题，rename后文件少了？因为rename后的文件与之前文件名重合，导致新文件会替换老文件。
        # #https://blog.csdn.net/qq_25745703/article/details/80216590
        # https://www.cnblogs.com/lfri/p/10615472.html：
        #os.listdir有问题，
        '''
         os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表，
         但有个很明显的缺点，
         它的默认顺序不是有序的或者说不是通常的顺序（不知道用啥排的）。
         
          所以导致文件被覆盖问题，比如：我要把00000010重名为0000002,但是原始的000002还没有被重名呢。
          也就是原始000010不按规则先原始000002一步，进行了重命名。
          
          解决方法：sort排序，重命名按照sort来
        '''


# if __name__ == "__main__":
#     main()



