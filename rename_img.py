import os
import configparser


def rename_photo():
    os.chdir("/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/config")
    conf = configparser.ConfigParser()
    conf.read("conf.ini")
    ii = conf.getint('file_num', 'count')
    newname = "{:0>12}".format(ii)
    ii += 1
    conf.set("file_num", "count", str(ii))
    with open("conf.ini", "w+") as f:
        conf.write(f)
    return newname

def main():
    rename_photo()


if __name__ == '__main__':
    main()


