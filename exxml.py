import xmltodict
from pathlib import Path
from xml.etree import ElementTree
save_path = '/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/test/'
xml_path = '/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/all_xml/000000000000.xml'


def xml_json(xml_path):
    xml_file = open(xml_path, 'r')
    xml_str = xml_file.read()
    json = xmltodict.parse(xml_str)
    print('json', json)
    return json


def json_xml(file_path, json_str):
    file_path = Path(file_path)
    annotation = xmltodict.unparse(json_str)
    xml_file_name = save_path + (file_path.name.split('.')[0] + '.xml')
    with open(xml_file_name, 'a') as f:
        f.write(annotation)


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作



if __name__ == '__main__':
    # json_xml(xml_path, xml_json(xml_path))
    tree = ElementTree.parse('/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/test/000000000000.xml')  # 解析movies.xml这个文件
    root = tree.getroot()  # 得到根元素，Element类
    pretty_xml(root, '\t', '\n')  # 执行美化方法
    tree.write('/home/hualai/ponder/lxy_project/EfficientDet-Pytorch/test/000000000000.xml')
