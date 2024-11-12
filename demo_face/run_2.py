# -*- coding:utf-8 -*-
import cv2
import time
import numpy as np
import multiprocessing as mp
from retinaface import Retinaface
from settings import cam_addres, img_shape


# 捕获视频流
def push_image(raw_q, cam_addr):
    cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
    # cap = cv2.VideoCapture(cam_addr)
    while True:
        t1 = time.time()
        is_opened, frame = cap.read()

        if is_opened:
            raw_q.put((frame, cam_addr, t1))
        else:
            cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
            # cap = cv2.VideoCapture(cam_addr)
        if raw_q.qsize() > 2:
            # 删除旧图片
            raw_q.get()
        else:
            # 等待
            time.sleep(0.01)


# 将原始队列中图像进行预测后放入预测队列
def predict(raw_q, pred_q):
    retinaface = Retinaface()

    while True:
        # t1 = time.time()
        # 读取某一帧
        raw_img, cam_address, t1 = raw_q.get()

        # 格式转变，BGRtoRGB
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        # 进行检测
        pred_img = np.array(retinaface.detect_image(raw_img, 'output/result', t1, cam_address))
        # RGBtoBGR满足opencv显示格式
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        pred_q.put(pred_img)
        toc = time.time()
        time_cost = toc - t1
        print('Processed in %.3fs FPS = %.3f' % (time_cost, 1 / time_cost))


def pop_image(pred_q, window_name, img_shape):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = pred_q.get()
        frame = cv2.resize(frame, img_shape)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


# 显示
def display(cam_addrs, window_names, img_shape=(300, 300)):
    raw_queues = [mp.Queue(maxsize=2) for _ in cam_addrs]
    pred_queues = [mp.Queue(maxsize=6) for _ in cam_addrs]
    processes = []

    for raw_q, pred_q, cam_addr, window_name in zip(raw_queues, pred_queues, cam_addrs, window_names):
        processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
        processes.append(mp.Process(target=predict, args=(raw_q, pred_q)))
        processes.append(mp.Process(target=pop_image, args=(pred_q, window_name, img_shape)))

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    mp.set_start_method(method='spawn')
    num_cameras = len(cam_addres)
    display(cam_addres[:num_cameras], ['camera' for _ in cam_addres], img_shape)

