import base64
from flask import request, Flask
import os
import xmltodict
import json
import shutil

app = Flask(__name__)
save_xml_path = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/all_xml1/'
save_img_path = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/all_image1/'
return_image_path = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/1203photo'
return_xml_path = '/home/hualai/ponder/12_samba_share/hualai_image_label_project/1203xml'
img_name_list = os.listdir(return_image_path)
img_name_list.sort()


@app.route("/xml", methods=['POST', 'GET'])
def get_xml():
    if request.method == 'POST':  # 客户端每点击一次next，将已标注后的xml传给服务端并返回新的xml与图片
        image_path = os.path.join(return_image_path, img_name_list[0])
        xml_path = os.path.join(return_xml_path, img_name_list[0].split('.')[0] + '.xml')
        img_name_list.pop(0)
        json_str = request.get_data()
        json_str = json.loads(json_str)
        image_name = json_str['annotation']['filename']
        with open(save_xml_path + (image_name.split('.')[0] + '.xml'), 'w') as obtain_xml:
            xmltodict.unparse(json_str, output=obtain_xml, pretty=True)
        with open(image_path, 'rb') as send_image:
            res = base64.b64encode(send_image.read())
        with open(xml_path, 'r') as send_xml:
            xml_str = send_xml.read()
            json_data = xmltodict.parse(xml_str)
            json_data['photo'] = str(res, 'utf-8')
            json.dumps(json_data)
            return json_data
    if request.method == 'GET':  # 初次登录时客户端请求为GET方法，向客户端返回一张图片及所对应的xml文件
        image_path = os.path.join(return_image_path, img_name_list[0])
        xml_path = os.path.join(return_xml_path, img_name_list[0].split('.')[0] + '.xml')
        img_name_list.pop(0)
        # print(img_name_list)
        with open(image_path, 'rb') as f:
            res = base64.b64encode(f.read())
        with open(xml_path, 'r') as xml_file:
            xml_str = xml_file.read()
            json_data = xmltodict.parse(xml_str)
            json_data['photo'] = str(res, 'utf-8')
            json.dumps(json_data)
        return json_data
    else:
        return 'Only receive POST and GET method!'


@app.route("/peterxli", methods=['POST', 'GET'])
def lxy_xml():
    if request.method == 'POST':  # 客户端每点击一次next，将已标注后的xml传给服务端
        json_str = request.get_data()
        json_str = json.loads(json_str)
        image_name = json_str['annotation']['filename']
        with open(save_xml_path + (image_name.split('.')[0] + '.xml'), 'w') as obtain_xml:
            xmltodict.unparse(json_str, output=obtain_xml, pretty=True)
            return 'ok'

    if request.method == 'GET':    # 初次登录时客户端请求为GET方法，向客户端返回一张图片及所对应的xml文件
        image_path = os.path.join(return_image_path, img_name_list[0])
        xml_path = os.path.join(return_xml_path, img_name_list[0].split('.')[0] + '.xml')
        img_name_list.pop(0)
        with open(image_path, 'rb') as f:
            res = base64.b64encode(f.read())
        with open(xml_path, 'r') as xml_file:
            xml_str = xml_file.read()
            json_data = xmltodict.parse(xml_str)
            json_data['photo'] = str(res, 'utf-8')
            json.dumps(json_data)
        shutil.move(image_path, save_img_path)
        shutil.move(xml_path, save_xml_path)
        return json_data
    else:
        return 'Only receive POST and GET method!'


@app.route('/hello')
def hello():
    return "hello world!"


if __name__ == "__main__":
    path=("/home/hualai/ponder/12_samba_share/hualai_image_label_project/")
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path, file)
            if (os.path.isdir(m)):
                h = os.path.split(m)
                list.append(h[1])
        print(list)
    app.run(host='0.0.0.0', port=2000)
