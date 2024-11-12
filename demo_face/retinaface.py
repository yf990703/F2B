import base64
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
from demo_face.nets_retinaface.retinaface import RetinaFace
from demo_face.util.anchors import Anchors
from demo_face.util.config import cfg_mnet, cfg_re50
from demo_face.util.utils2 import (Alignment_1, compare_faces, letterbox_image,
                                   preprocess_input)
from demo_face.util.utils_bbox import (decode, decode_landm, non_max_suppression,
                                       retinaface_correct_boxes)
import requests
import json
import urllib.parse
# from settings import user_data
# from torch2trt import TRTModule


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
        "retinaface_model_path": 'D:/runcode/Yolov5-Deepsort-Fastreid-main/demo_face/model/Retinaface_resnet50.pth',
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
        "Re_model_path": 'D:/runcode/Yolov5-Deepsort-Fastreid-main/demo_face/model/backbone_ir50_asia.pth',
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
        "threhold": 70,
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
        self.net = None
        self.model = None
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
        
        # trt #
        # trt_retinaface_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/retinaface_model_trt.pth'
        # trt_ir_50_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/ir_50_model_trt.pth'
        trt_retinaface_model_path = ''
        trt_ir_50_model_path = ''
        self.trt_retinaface_model_path = trt_retinaface_model_path
        self.trt_ir_50_model_path = trt_ir_50_model_path


    def load_face_features(self, encoding):
        # try:
        #     self.known_face_encodings = np.load(
        #         f"demo_face/model/{self.backbone}_face_encoding.npy")
        #     self.known_face_names = np.load(
        #         f"demo_face/model/{self.backbone}_names.npy")
        # except:
        #     if not encoding:
        #         print("载入已有人脸特征失败，请检查model下面是否生成了相关的人脸特征文件。")
        try:
            self.known_face_encodings = np.load(f"demo_face/model/{self.backbone}_face_encoding.npy")
            self.known_face_names = np.load(f"demo_face/model/{self.backbone}_names.npy")
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model下面是否生成了相关的人脸特征文件。")
        else:
            if not hasattr(self, 'known_face_encodings') or self.known_face_encodings is None:
                self.known_face_encodings = np.load(f"demo_face/model/{self.backbone}_face_encoding.npy")
                self.known_face_names = np.load(f"demo_face/model/{self.backbone}_names.npy")

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def load_detection_model(self):
        # -------------------------------#
        #   载入检测模型与权值
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
        print('检测模型加载完成!')


    def load_comparison_model(self):
        # -------------------------------#
        #   载入比对模型与权值
        # -------------------------------#
        device = torch.device('cuda' if self.cuda else 'cpu')
        if self.trt_ir_50_model_path:
            self.model = TRTModule()
            self.model.load_state_dict(torch.load(self.trt_ir_50_model_path, map_location=device))
            print('1')
        else:
            self.model = IR_50([112, 112]).eval()
            state_dict = torch.load(self.Re_model_path, map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            if self.cuda:
                if torch.cuda.is_available():
                    print("CUDA is available!")
                else:
                    print("CUDA is not available.")
                self.model = nn.DataParallel(self.model)
                self.model = self.model.cuda()
        print('比对模型加载完成!')
    
    def encode_face_dataset(self, image_paths, names):

        if self.net is None:
            self.load_detection_model()

        if self.model is None:
            self.load_comparison_model()

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

        np.save("demo_face/model/{backbone}_face_encoding.npy".format(backbone=self.backbone), face_encodings)
        np.save("demo_face/model/{backbone}_names.npy".format(backbone=self.backbone), known_names)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------

    def detect_image(self, image, timestamp):
        if self.net is None:
            self.load_detection_model()

        if self.model is None:
            self.load_comparison_model()
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        results = []
        face_result = []
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
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return results

            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms,
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

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

        face_names = []
        self.load_face_features(encoding=True)
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
            b = list(map(int, b))
            name = face_names[i]
            if name != "Unknown":
                x1 = max(0, b[0])
                y1 = max(0, b[1])
                x2 = min(old_image.shape[1], b[2])
                y2 = min(old_image.shape[0], b[3])

                if x2 > x1 and y2 > y1:
                    crop_img = old_image[y1:y2, x1:x2]
                    _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    base64_encoded_img = base64.b64encode(buffer).decode('utf-8')

                    x1 = float(b[0])
                    y1 = float(b[1])
                    x2 = float(b[2])
                    y2 = float(b[3])
                    w = x2 - x1
                    h = y2 - y1

                    face_result = [x1, y1, x2, y2]
                    results.append(face_result)

                    # result_dict = {
                    #     "time": timestamp,
                    #     "tid": name,
                    #     "x": x1,
                    #     "y": y1,
                    #     "w": w,
                    #     "h": h,
                    #     "score": float(best_score),
                    #     "image_tid": base64_encoded_img
                    # }

                    # results.append(result_dict)

        return results

    def photograph(self, image, timestamp):
        results = []
        old_image = image.copy()
        image = np.array(image, np.float32)
        # ---------------------------------------------------#
        #   Retinaface检测部分-开始
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
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return results

            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms,
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

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
            crop = crop_img
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

        for i, b in enumerate(boxes_conf_landms):
            b = list(map(int, b))
            x1 = max(0, b[0])
            y1 = max(0, b[1])
            x2 = min(old_image.shape[1], b[2])
            y2 = min(old_image.shape[0], b[3])

            if x2 > x1 and y2 > y1:
                _, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                base64_encoded_img = base64.b64encode(buffer).decode('utf-8')

                x1 = float(b[0])
                y1 = float(b[1])
                x2 = float(b[2])
                y2 = float(b[3])
                w = x2 - x1
                h = y2 - y1

                result_dict = {
                    "time": timestamp,
                    "tid": i,
                    "x": x1,
                    "y": y1,
                    "w": w,
                    "h": h,
                    "encoding": face_encodings[i].tolist(),
                    "image_tid": base64_encoded_img
                }

                results.append(result_dict)

        return results


    def compare_face_to_images(self, reference_image, image_list, threshold=0.5):
        """
        比对一张已知的人脸图片与一系列图片，返回相似度高于阈值的匹配结果。
        :param reference_image: 已知的人脸图片（numpy数组）
        :param image_list: 要比对的图片列表（numpy数组的列表）
        :param threshold: 相似度阈值，越大表示越相似
        :return: 匹配结果列表，每个元素是一个包含(图片索引, 相似度得分)的元组
        """
        # 确保比对模型已加载
        if self.model is None:
            self.load_comparison_model()
        
        reference_detections = self.photograph(reference_image, "timestamp_not_used")
        if not reference_detections:
            raise ValueError("No face detected in the reference image.")
        reference_encoding = reference_detections[0]["encoding"]
        
        matches = []
        for idx, image in enumerate(image_list):
            detections = self.photograph(image, "timestamp_not_used")
            for detection in detections:
                face_encoding = detection["encoding"]
                similarity_score = 1 - cosine(reference_encoding, face_encoding)
                if similarity_score > threshold:
                    matches.append((idx, similarity_score))

        return matches
