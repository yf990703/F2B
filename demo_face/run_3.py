# -*- coding:utf-8 -*-
import cv2
import time
import numpy as np
import multiprocessing as mp
from retinaface import Retinaface
from settings import cam_addres, img_shape
import ffmpeg
import logging
# 捕获视频流
# def push_image(raw_q, cam_addr):
#     cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
#     while True:
#         t1 = time.time()
#         is_opened, frame = cap.read()
#
#         if is_opened:
#             raw_q.put((frame, cam_addr, t1))
#         else:
#             cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
#         if raw_q.qsize() > 2:
#             # 删除旧图片
#             raw_q.get()
#         else:
#             # 等待
#             time.sleep(0.01)


def detect_stream_decoder(source):
    probe = ffmpeg.probe(source)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    codec_name = video_info['codec_name']

    if codec_name == 'h264':
        return "h264_cuvid"
    elif codec_name == 'hevc' or codec_name == 'h265':
        return "hevc_cuvid"
    else:
        raise ValueError(f'Unsupported codec: {codec_name}')

def push_image(raw_q, source):
    decoder = detect_stream_decoder(source)
    args = {
        "rtsp_transport": "tcp",
        "fflags": "nobuffer",
        "flags": "low_delay",
        "hwaccel": "cuda",
        "c:v": decoder
    }

    probe = ffmpeg.probe(source)
    cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    print("fps: {}".format(cap_info['r_frame_rate']))
    width = cap_info['width']
    height = cap_info['height']
    up, down = str(cap_info['r_frame_rate']).split('/')
    fps = eval(up) / eval(down)
    print("fps: {}".format(fps))

    process1 = (
        ffmpeg
        .input(source, **args)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )


    while True:
        t1 = time.time()
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break

        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )

        if raw_q.qsize() > 5:
            raw_q.get()

        logging.debug('Pushing frame into queue. Queue size: {}'.format(raw_q.qsize()))
        raw_q.put((in_frame, source, t1))




def predict(raw_q, pred_q):
    retinaface = Retinaface()
    frame_count = 0
    while True:

        raw_img, cam_address, t1 = raw_q.get()

        t2 = time.time()
        pred_img = np.array(retinaface.detect_image(raw_img, 'output/result', t1, cam_address))
        # RGBtoBGR满足opencv显示格式
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        pred_q.put(pred_img)
        toc = time.time()
        time_cost = toc - t2
        frame_count += 1
        print('Processed in %.3fs FPS = %.3f' % (time_cost, 1 / time_cost))


def pop_image(pred_q, window_name, img_shape):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = pred_q.get()
        # frame = cv2.resize(frame, img_shape)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


# 显示
def display(cam_addrs, window_names, img_shape=(300, 300)):
    raw_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]
    pred_queues = [mp.Queue(maxsize=3) for _ in cam_addrs]
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

