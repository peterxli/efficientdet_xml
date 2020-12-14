import os
import shutil

your_jpg_path = r'/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/test'
# temp_jpg_path = r'\\192.168.107.58\share\hualai_image_label_project\temp_image'
total = len(os.listdir(your_jpg_path))
print("total : ", total)
filelist = os.listdir(your_jpg_path)

'''
更新，解决filelist无序问题
'''
filelist.sort(key=lambda x: int(x[:-4]))  ##倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大排序

# count = 0
ii = 0

num_list = []
for img in filelist:

    print("img", img)


    if ii < ii+total:
        newname = "{:0>12}".format(ii)
        print("newname : ", newname)
        ii += 1

        src = os.path.join(os.path.abspath(your_jpg_path), img)
        # print("src : ", src)
        dst = os.path.join(os.path.abspath(your_jpg_path), newname + '.jpg')
        # print("dst : " ,dst)
        os.rename(src, dst)
        # shutil.copy(src, dst)




