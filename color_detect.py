import cv2


def crop_img():

    img = cv2.imread('/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/photo/000000000000.jpg', 1)
    crop = img[0:972, 0:1920]
    cv2.imshow('imshow', crop)
    cv2.imwrite('/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/photo/000000000000.jpg', crop)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


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


if __name__ == '__main__':
    crop_img()