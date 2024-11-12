#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/18
# @Author : zengwb

import os
import cv2
import numpy as np
import torch
import warnings
import argparse
import onnxruntime as ort
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes, draw_person
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from two_step.yolo import YOLO
from person_detect_yolov5 import Person_detect
from demo_face.retinaface import Retinaface
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from demo_face.retinaface import Retinaface
import time
from fast_reid.demo.person_bank import Reid_feature

class yolo_reid():
    def __init__(self, cfg, args, path):
        self.yolo = YOLO()
        self.retinaface = Retinaface()
        self.reid_feature = Reid_feature
        self.args = args
        self.video_path = path
        self.output_dir = args.output_dir
        self.model_loaded = False
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        # Person_detect行人检测类
        self.person_detect = Person_detect(self.args, self.video_path)
        # deepsort 类
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)
        imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        self.dataset = LoadImages(self.video_path, img_size=imgsz)
        # self.query_feat = np.load(path)
        # self.names = np.load(args.names)

    def deep_sort(self):
        folder_path = 'D:/runcode/Yolov5-Deepsort-Fastreid-main/fast_reid/query/a1111111'
        output_folder = 'D:/resultsavepic/xmb/'
        idx_frame = 0
        results = []
        retinaface_ec = Retinaface(1)
        list_dir = os.listdir("demo_face/face")
        image_paths = []
        names = []
        show_img = []
        # for name in list_dir:
        #     image_paths.append("demo_face/face/" + name)
        #     name_without_extension = os.path.splitext(name)[0]
        #     names.append(name_without_extension)
        #
        # retinaface_ec.encode_face_dataset(image_paths, names)

        for video_path, img, ori_img, vid_cap in self.dataset:
            idx_frame += 1
            ori_pic = ori_img
            save_pictest = [] # shanchu
            ori_fac = ori_pic
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
            # t1 = time_synchronized()
            # retinaface_ec = Retinaface(1)
            # list_dir = os.listdir("demo_face/face")
            # image_paths = []
            # names = []
            # for name in list_dir:
            #     image_paths.append("demo_face/face/" + name)
            #     name_without_extension = os.path.splitext(name)[0]
            #     names.append(name_without_extension)

            # retinaface_ec.encode_face_dataset(image_paths, names)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            # 转变成Image
            ori_img = Image.fromarray(np.uint8(ori_img))
            # yolo facedetection
            t2 = time.time()
            ###########################################################################

            # face_xywh = self.retinaface.detect_image(ori_pic, t2)
            #######################################################################

            ####################################
            # for i, b in enumerate(face_xywh):
            #     b = list(map(int, b))
            #     # if name != "Unknown":
            #     x1 = max(0, b[0])
            #     y1 = max(0, b[1])
            #     x2 = min(ori_pic.shape[1], b[2])
            #     y2 = min(ori_pic.shape[0], b[3])
            #     if x2 > x1 and y2 > y1:
            #         crop_img = ori_pic[y1:y2, x1:x2]
            #         save_pictest.append(crop_img)
            #
            # for i in save_pictest:
            #     i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            #     filename = f'foundfaceyf.jpg'
            #     # Construct the full path to save the image
            #     output_path = os.path.join(output_folder, filename)
            #     # Save the image to the specified path
            #     cv2.imwrite(output_path, i)
            #     cv2.imshow('111', i)
            #     cv2.waitKey()
            ###########################################################################################
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)
            # bbox_xywh, cls_conf, cls_ids, xy = self.yolo.detect_image(ori_img)
            print(bbox_xywh, cls_conf, cls_ids, xy)
            # self.yolo.matching_body(face_xywh, xy, ori_pic)
            ######################################################################################################
            # if not os.listdir(folder_path):
            #     continue
            # if not self.model_loaded:
            #     path_features, path_names = self.reid_feature.face_feature_extraction(self)
            #     query_feat = np.load(path_features)
            #     names = np.load(path_names)
            #     self.model_loaded = True
            ###############################################################################################
            path_features, path_names = self.reid_feature.face_feature_extraction(self)
            query_feat = np.load(path_features)
            names = np.load(path_names)

            # do tracking
            outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_pic)
            print(len(outputs), len(bbox_xywh), features.shape)

            person_cossim = cosine_similarity(features, query_feat)
            max_idx = np.argmax(person_cossim, axis=1)
            maximum = np.max(person_cossim, axis=1)
            max_idx[maximum < 0.6] = -1
            score = maximum
            reid_results = max_idx
            draw_person(ori_pic, xy, reid_results, names)  # draw_person name



            # print(features.shape, self.query_feat.shape, person_cossim.shape, features[1].shape)

            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_pic, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
###############################################################################################################################
            # 保存实验结果图用
            # output_path = os.path.join(self.output_dir, f"chinese_arc_yyjlk1.jpg")  # Add this line
            # cv2.imwrite(output_path, ori_pic)
###############################################################################################################################
                # results.append((idx_frame - 1, bbox_tlwh, identities))
            # print("yolo+deepsort:", time_synchronized() - t1)

            if self.args.display:
                cv2.imshow("test", ori_pic)
                cv2.waitKey()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='D:/results_display/testvideo.mp4', type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')  # parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=False, help='True: sort model or False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--output_dir", type=str, default="D:/resultsavepic", help="directory to save the output results")

    # reid
    # parser.add_argument("--query", type=str, default="./fast_reid/query/query_features.npy")
    # parser.add_argument("--names", type=str, default="./fast_reid/query/names.npy")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    yolo_reid = yolo_reid(cfg, args, path=args.video_path)
    with torch.no_grad():
        yolo_reid.deep_sort()
