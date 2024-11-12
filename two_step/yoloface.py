import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
import cv2 as cv2
from nets.yolo import YoloBody
from two_step.utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from two_step.utils.utils_bbox import decode_outputs, non_max_suppression


def overlap_ratio(box1, box2):
    # 计算交集区域
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算较小框的面积
    min_area = min(area1, area2)

    # 计算重叠比例
    if min_area == 0:
        return 0
    else:
        return inter_area / min_area


def match_face_body(face_boxes, body_boxes, overlap_threshold=0.90):
    matched_pairs = []
    for face_box in face_boxes:
        for body_box in body_boxes:
            overlap = overlap_ratio(face_box, body_box)
            if overlap >= overlap_threshold:
                matched_pairs.append((face_box, body_box))
                break
    return matched_pairs


def create_combined_boxes(matched_pairs):
    combined_boxes = []
    for face_box, body_box in matched_pairs:
        combined_box = [
            min(face_box[0], body_box[0]),  # x1
            min(face_box[1], body_box[1]),  # y1
            max(face_box[2], body_box[2]),  # x2
            max(face_box[3], body_box[3])  # y2
        ]
        combined_boxes.append(combined_box)
    return combined_boxes

def creat_center_wh(x1, y1, x2, y2):
    obj = []
    x_c = (x1 + x2) / 2  # 中心点x坐标
    y_c = (y1 + y2) / 2  # 中心点y坐标
    w = abs(x2 - x1)  # 宽
    h = abs(y2 - y1)  # 高
    obj = [x_c, y_c, w, h]
    return obj

class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'D:/runcode/Yolov5-Deepsort-Fastreid-main/two_step/model_data/best_epoch_weights.pth',
        "classes_path": 'D:/runcode/Yolov5-Deepsort-Fastreid-main/two_step/model_data/cls_classes.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        # ---------------------------------------------------------------------#
        "phi": 's',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.4,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = YoloBody(self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        bbox_xywh = []
        cls_conf = []
        cls_ids = []
        # cv2.imshow('111', image)
        # cv2.waitKey()
        save_pic2 = [] # 每一帧图的检测对象
        old_frame = np.array(image)
        old_frame2 = image
        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
            # print(top_label,top_conf,top_boxes)
            # cls_ids.extend(top_label.tolist())  # 将top_label数组中的值存入cls_ids列表
            # for label in top_label:
            #     if label == 1:
            #         cls_ids.append(label)
            # cls_conf.extend(top_conf.tolist())  # 将top_conf数组中的值存入cls_conf列表
            # 遍历每个检测框的标签和置信度
            for label, conf in zip(top_label, top_conf):
                if label == 1:  # 仅当类别为人时
                    cls_ids.append(label)
                    cls_conf.append(conf)

            face_boxes = []
            body_boxes = []
            for i, label in enumerate(top_label):
                if self.class_names[label] == 'face':
                    face_boxes.append(top_boxes[i])
                elif self.class_names[label] == 'person':
                    body_boxes.append(top_boxes[i])

        matched_pairs = match_face_body(face_boxes, body_boxes)
        # print(face_boxes, body_boxes)
        combined_boxes = create_combined_boxes(matched_pairs)

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        #   返回框中的每个小人的图像                                   #
        # for i, b in enumerate(body_boxes):
        #     b = list(map(int, b))
        #     # if name != "Unknown":
        #     x1 = max(0, b[1])
        #     y1 = max(0, b[0])
        #     x2 = min(old_frame.shape[1], b[3])
        #     y2 = min(old_frame.shape[0], b[2])
        #     if x2 > x1 and y2 > y1:
        #         crop_img = old_frame[y1:y2, x1:x2]
        #         save_pic2.append(crop_img)
        # return save_pic2
        # ----------------------------------------------------------#
        for i, b in enumerate(body_boxes):
            b = list(map(int, b))
            # if name != "Unknown":
            x1 = max(0, b[1])
            y1 = max(0, b[0])
            x2 = min(old_frame.shape[1], b[3])
            y2 = min(old_frame.shape[0], b[2])
            if x2 > x1 and y2 > y1:
                bbox_xywh.append(creat_center_wh(x1, y1, x2, y2))
        #
        # for i, c in list(enumerate(top_label)):
        #     predicted_class = self.class_names[int(c)]
        #     box = top_boxes[i]
        #     score = top_conf[i]
        #
        #     top, left, bottom, right = box
        #
        #     top = max(0, np.floor(top).astype('int32'))
        #     left = max(0, np.floor(left).astype('int32'))
        #     bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        #     right = min(image.size[0], np.floor(right).astype('int32'))
        #
        #     label = '{} {:.2f}'.format(predicted_class, score)
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)
        #     label = label.encode('utf-8')
        #     # print(label, top, left, bottom, right)
        #
        #     # 下面是框脸
        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])
        #
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
        #     ######################################################################################
        #
        #     # del draw
        #
        # for box in combined_boxes:
        #     top, left, bottom, right = box
        #     top = max(0, np.floor(top).astype('int32'))
        #     left = max(0, np.floor(left).astype('int32'))
        #     bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        #     right = min(image.size[0], np.floor(right).astype('int32'))
        #
        #     combined_box_color = (255, 0, 0)
        #
        #     draw = ImageDraw.Draw(image)
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=combined_box_color)
        #
        #     del draw

        return np.array(bbox_xywh), cls_conf, cls_ids, body_boxes  # np.array(bbox_xywh)

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                # ---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                # ---------------------------------------------------------#
                results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)

        outputs = [output.cpu().numpy() for output in outputs]
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200)
        print("Save to the " + heatmap_save_path)
        plt.cla()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
