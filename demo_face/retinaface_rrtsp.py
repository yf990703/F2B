import os
import time
import datetime
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from demo_face.backbone.model_irse import IR_50
from nets_retinaface.retinaface import RetinaFace
from util.anchors import Anchors
from util.config import cfg_mnet, cfg_re50
from util.utils2 import (Alignment_1, compare_faces, letterbox_image,
                        preprocess_input)
from util.utils_bbox import (decode, decode_landm, non_max_suppression,
                             retinaface_correct_boxes)
import requests
import json
import urllib.parse
# from settings import user_data
from torch2trt import TRTModule


def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # ---------------#
    #   设置字体
    # ---------------#
    font = ImageFont.truetype(font='model/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)


#  注意backbone和model_path的对应
class Retinaface(object):
    _defaults = {
        # ----------------------------------------------------------------------#
        #   retinaface训练完的权值路径
        # ----------------------------------------------------------------------#
        "retinaface_model_path": '/home/admin2/PycharmProjects/pythonProject/demo_face/model/Retinaface_resnet50.pth',
        # ----------------------------------------------------------------------#
        #   retinaface所使用的主干网络，有mobilenet和resnet50
        # ----------------------------------------------------------------------#
        "retinaface_backbone": "resnet50",
        # ----------------------------------------------------------------------#
        #   retinaface中只有得分大于置信度的预测框会被保留下来
        # ----------------------------------------------------------------------#
        "confidence": 0.25,
        # ----------------------------------------------------------------------#
        #   retinaface中非极大抑制所用到的nms_iou大小
        # ----------------------------------------------------------------------#
        "nms_iou": 0.4,
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        # ----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        # ----------------------------------------------------------------------#
        "letterbox_image": True,

        # ----------------------------------------------------------------------#
        #   训练完的权值路径
        # ----------------------------------------------------------------------#
        "Re_model_path": '/home/admin2/PycharmProjects/pythonProject/demo_face/model/backbone_ir50_asia.pth',
        # ----------------------------------------------------------------------#
        #   使用的主干网络
        # ----------------------------------------------------------------------#
        "backbone": "Ir_50",
        # ----------------------------------------------------------------------#
        #  输入图片大小
        # ----------------------------------------------------------------------#
        "Re_input_shape": [112, 112],
        # ----------------------------------------------------------------------#
        #   阈值
        # ----------------------------------------------------------------------#
        "threhold": 50,
        # --------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Retinaface
    # ---------------------------------------------------#
    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # trt #
        # trt_retinaface_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/retinaface_model_trt.pth'
        # trt_ir_50_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/ir_50_model_trt.pth'
        trt_retinaface_model_path = ''
        trt_ir_50_model_path = ''
        self.trt_retinaface_model_path = trt_retinaface_model_path
        self.trt_ir_50_model_path = trt_ir_50_model_path
        # ---------------------------------------------------#
        #   不同主干网络的config信息
        # ---------------------------------------------------#
        if self.retinaface_backbone == "resnet50":
            self.cfg = cfg_re50
        else:
            self.cfg = cfg_mnet

        # ---------------------------------------------------#
        #   先验框的生成
        # ---------------------------------------------------#
        self.anchors = Anchors(self.cfg, image_size=(
            self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        self.generate()

        try:
            self.known_face_encodings = np.load(
                "model/{backbone}_face_encoding.npy".format(backbone=self.backbone))
            self.known_face_names = np.load("model/{backbone}_names.npy".format(backbone=self.backbone))
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model下面是否生成了相关的人脸特征文件。")
            pass

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        device = torch.device('cuda' if self.cuda else 'cpu')
        if self.trt_retinaface_model_path:
            self.net = TRTModule()
            self.net.load_state_dict(torch.load(self.trt_retinaface_model_path, map_location=device))
        else:
            self.net = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
            state_dict = torch.load(self.retinaface_model_path, map_location=device)
            self.net.load_state_dict(state_dict)
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

        if self.trt_ir_50_model_path:
            self.model = TRTModule()
            self.model.load_state_dict(torch.load(self.trt_ir_50_model_path, map_location=device))
        else:
            self.model = IR_50([112, 112]).eval()
            state_dict = torch.load(self.Re_model_path, map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            if self.cuda:
                self.model = nn.DataParallel(self.model)
                self.model = self.model.cuda()
                print('normal')

        print('Finished!')

    def encode_face_dataset(self, image_paths, names):

        encoding_file = 'model/{backbone}_face_encoding.npy'.format(backbone=self.backbone)
        names_file = 'model/{backbone}_names.npy'.format(backbone=self.backbone)

        if os.path.isfile(encoding_file) and os.path.isfile(names_file):
            face_encodings = np.load(encoding_file, allow_pickle=True).tolist()
            known_names = np.load(names_file, allow_pickle=True).tolist()
            for name in names:
                known_names.append(name)
        else:
            face_encodings = []
            known_names = names

        for index, path in enumerate(tqdm(image_paths)):
            # ---------------------------------------------------#
            #   打开人脸图片
            # ---------------------------------------------------#
            image = np.array(Image.open(path), np.float32)
            # 转换成3通道
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            # print(path)
            # ---------------------------------------------------#
            #   对输入图像进行一个备份
            # ---------------------------------------------------#
            old_image = image.copy()
            # ---------------------------------------------------#
            #   计算输入图片的高和宽
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            # ---------------------------------------------------#
            #   计算scale，用于将获得的预测框转换成原图的高宽
            # ---------------------------------------------------#
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            # ---------------------------------------------------#
            #   将处理完的图片传入Retinaface网络当中进行预测
            # ---------------------------------------------------#
            with torch.no_grad():
                # -----------------------------------------------------------#
                #   图片预处理，归一化。
                # -----------------------------------------------------------#
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
                    torch.FloatTensor)

                if self.cuda:
                    image = image.cuda()
                    anchors = anchors.cuda()

                loc, conf, landms = self.net(image)
                # -----------------------------------------------------------#
                #   对预测框进行解码
                # -----------------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                # -----------------------------------------------------------#
                #   获得预测结果的置信度
                # -----------------------------------------------------------#
                conf = conf.data.squeeze(0)[:, 1:2]
                # -----------------------------------------------------------#
                #   对人脸关键点进行解码
                # -----------------------------------------------------------#
                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   对人脸检测结果进行堆叠
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

                if len(boxes_conf_landms) <= 0:
                    print(known_names[index], "：未检测到人脸")
                    continue
                # ---------------------------------------------------------#
                #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.retinaface_input_shape[0],
                                                                           self.retinaface_input_shape[1]]),
                                                                 np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            # ---------------------------------------------------#
            #   选取最大的人脸框。
            # ---------------------------------------------------#
            best_face_location = None
            biggest_area = 0
            for result in boxes_conf_landms:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face_location = result

            # ---------------------------------------------------#
            #   截取图像
            # ---------------------------------------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                       int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
                [int(best_face_location[0]), int(best_face_location[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.Re_input_shape[1], self.Re_input_shape[0]))) / 255
            crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img, 0)
            # ---------------------------------------------------#
            #   利用图像算取特征向量
            # ---------------------------------------------------#

            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                face_encoding = self.model(crop_img)[0].cpu().numpy()
                # print(face_encoding)
                face_encodings.append(face_encoding)
                # print(face_encodings)

        np.save("model/{backbone}_face_encoding.npy".format(backbone=self.backbone), face_encodings)
        np.save("model/{backbone}_names.npy".format(backbone=self.backbone), known_names)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, txt_path, now_time, cam_addrs):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_image = image.copy()
        # ---------------------------------------------------#
        #   把图像转换成numpy的形式
        # ---------------------------------------------------#
        image = np.array(image, np.float32)
        # ---------------------------------------------------#
        #   Retinaface检测部分-开始
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # ---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # ---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        # ---------------------------------------------------#
        with torch.no_grad():
            # -----------------------------------------------------------#
            #   图片预处理，归一化。
            # -----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image = image.cuda()
            # ---------------------------------------------------------#
            #   传入网络进行预测
            # ---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            # ---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf = conf.data.squeeze(0)[:, 1:2]

            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   对人脸检测结果进行堆叠
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            # ---------------------------------------------------#
            #   如果没有预测框则返回原图
            # ---------------------------------------------------#
            if len(boxes_conf_landms) <= 0:
                return old_image

            # ---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        # ---------------------------------------------------#
        #   Retinaface检测部分-结束
        # ---------------------------------------------------#

        # -----------------------------------------------#
        #   人脸识别编码部分-开始
        # -----------------------------------------------#
        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:
            # ----------------------#
            #   图像截取，人脸矫正
            # ----------------------#
            boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
            crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                       int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            # ----------------------#
            #   人脸编码
            # ----------------------#
            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.Re_input_shape[1], self.Re_input_shape[0]))) / 255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()
                face_encoding = self.model(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        # -----------------------------------------------#
        #  编码部分-结束
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   人脸特征比对-开始
        # -----------------------------------------------#
        face_names = []
        for face_encoding in face_encodings:
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            matches, face_similarities = compare_faces(self.known_face_encodings, face_encoding,
                                                       tolerance=self.threhold)
            name = "Unknown"

            #   取出这个最近人脸的评分
            #   取出当前输入进来的人脸，最接近的已知人脸的序号
            best_match_index = np.argmax(face_similarities)
            best_score = face_similarities[best_match_index]

            if best_score >= self.threhold and matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # -----------------------------------------------#
        #   人脸特征比对-结束
        # -----------------------------------------------#

        for i, b in enumerate(boxes_conf_landms):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # ---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            # ---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] - 10
            im0 = old_image
            percentage_score = "{:.1f}%".format(best_score)
            cv2.putText(old_image, percentage_score, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # ---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            # ---------------------------------------------------#
            # cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            # cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            # cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            # cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            # cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

            name = face_names[i]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            # --------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
            # 保存识别后的人脸图像，姓名，坐标，置信度
            # cls = ['李新']
            dt = datetime.datetime.fromtimestamp(now_time)
            time = dt.strftime("%H:%M:%S")
            if name != "unknown":

                file_name = f'{str(name)}_{time}.jpg'
                file_path = os.path.join(txt_path, file_name)
                cv2.imwrite(file_path, old_image)

                # with open(txt_path + '.txt', 'a') as f:
                #     f.write(f'姓名为：{name}，左上角的座标为：({b[0]},{b[1]}),右下角的坐标为：({b[2]},{b[3]}),置信度为{str(best_score)}' + '\n')

                server_url = 'http://192.168.100.72:35610/location/'
                server_url_local = "http://0.0.0.0:8000"
                relative_path = os.path.join("output/run/result", file_name)
                image_url = os.path.join(server_url_local, relative_path)

                def find_phone_by_name(user_data, target_name):
                    for phone, info in user_data.items():
                        if info['name'] == target_name:
                            return phone
                    return None

                # num = find_phone_by_name(user_data, name)
                parsed_url = urlparse(cam_addrs)
                cam_ip = parsed_url.netloc.split('@')[-1]

                # cam_ip = '192.168.29.31'
                num = 13921319226
                type = 2
                # 定义数据
                data = {

                    "personID": str(name),
                    "ratio": str(best_score),
                    "cameraIP": str(cam_ip),
                    "timestamp": str(now_time),
                    "resultImage": str(image_url),
                    "phonenum": str(num),
                    "X1": int(b[0]),
                    "X2": int(b[2]),
                    "Y1": int(b[1]),
                    "Y2": int(b[3]),
                    'type': str(type)
                }
                print(data)
                # payload = json.dumps(data)
                # headers = {'Content-Type': 'application/json'}
                # requests.post(server_url, data=data)

        return old_image
