# from tkinter import N
# from tkinter.tix import Tree
import torch.multiprocessing as mp
# mp.set_start_method('forkserver', force=True)
import sys
import os
from typing import Dict
import time
# import pymysql
import sys
import sh
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# from main3me_nodatabase import parse_args1, detectMain

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path) and len(path) > 1:
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)
from retinaface_rtsp import Retinaface
import torch
import _PyNvCodec as nvc
import PytorchNvCodec as pnvc
import numpy as np
import cv2

from io import BytesIO
from torch.multiprocessing import Process, Queue, Pipe
import signal
import subprocess
import json


def get_stream_params(url: str) -> Dict:

    ffprobe_path = "/home/admin2/PycharmProjects/pythonProject/install/FFmpeg/build_x64_release_shared/bin/ffprobe"
    lib_path = "/home/admin2/PycharmProjects/pythonProject/install/FFmpeg/build_x64_release_shared/lib"
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        url
    ]
    while True:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            stdout = proc.communicate(timeout=10)[0]
            break
        except subprocess.TimeoutExpired:
            proc.kill()
            # proc.terminate()
            # os.killpg(proc.pid, signal.SIGTERM)
            continue
        except subprocess.TimeoutExpired:
            print("Command timeout expired, trying again.")

    bio = BytesIO(stdout)
    json_out = json.load(bio)

    # while True:
    #     try:
    #         output = sh.Command(ffprobe_path)(
    #             '-v', 'quiet',
    #             '-print_format', 'json',
    #             '-show_format',
    #             '-show_streams',
    #             url,
    #             _timeout=10,
    #             _env=os.environ  # Add this line
    #         )
    #
    #         json_out = json.loads(str(output))
    #         print(json_out)
    #         break
    #     except sh.TimeoutException:
    #         print("Command timeout expired, waiting for 10 seconds before retrying.")
    #         time.sleep(10)
    #     except sh.ErrorReturnCode as e:
    #         print(f"Command failed with error: {str(e)}, waiting for 10 seconds before retrying.")
    #         time.sleep(10)

    print(url)
    # print(json_out)
    params = {}
    if not 'streams' in json_out:
        return {}

    for stream in json_out['streams']:
        if stream['codec_type'] == 'video':
            params['width'] = stream['width']
            params['height'] = stream['height']

            codec_name = stream['codec_name']
            is_h264 = True if codec_name == 'h264' else False
            is_hevc = True if codec_name == 'hevc' else False
            if not is_h264 and not is_hevc:
                raise ValueError("Unsupported codec: " + codec_name +
                                 '. Only H.264 and HEVC are supported in this sample.')
            else:
                params['codec'] = nvc.CudaVideoCodec.H264 if is_h264 else nvc.CudaVideoCodec.HEVC

                pix_fmt = stream['pix_fmt']
                is_yuv420 = pix_fmt == 'yuv420p'
                is_yuv444 = pix_fmt == 'yuv444p'

                # YUVJ420P and YUVJ444P are deprecated but still wide spread, so handle
                # them as well. They also indicate JPEG color range.
                is_yuvj420 = pix_fmt == 'yuvj420p'
                is_yuvj444 = pix_fmt == 'yuvj444p'

                if is_yuvj420:
                    is_yuv420 = True
                    params['color_range'] = nvc.ColorRange.JPEG
                if is_yuvj444:
                    is_yuv444 = True
                    params['color_range'] = nvc.ColorRange.JPEG

                if not is_yuv420 and not is_yuv444:
                    raise ValueError("Unsupported pixel format: " +
                                     pix_fmt +
                                     '. Only YUV420 and YUV444 are supported in this sample.')
                else:
                    params['format'] = nvc.PixelFormat.NV12 if is_yuv420 else nvc.PixelFormat.YUV444

                # Color range default option. We may have set when parsing
                # pixel format, so check first.
                if 'color_range' not in params:
                    params['color_range'] = nvc.ColorRange.MPEG
                # Check actual value.
                if 'color_range' in stream:
                    color_range = stream['color_range']
                    if color_range == 'pc' or color_range == 'jpeg':
                        params['color_range'] = nvc.ColorRange.JPEG

                # Color space default option:
                params['color_space'] = nvc.ColorSpace.BT_601
                # Check actual value.
                if 'color_space' in stream:
                    color_space = stream['color_space']
                    if color_space == 'bt709':
                        params['color_space'] = nvc.ColorSpace.BT_709

                return params
    return {}


def rtsp_client(url, name, gpu_id, qu):
    # Get stream parameters
    params = get_stream_params(url)
    # if not os.path.exists('/ai35liyx/' + url.split('.')[-1]):
    #     os.makedirs('/ai35liyx/' + url.split('.')[-1])
    if not len(params):
        raise ValueError("Can not get " + url + ' streams params')

    w = params['width']
    h = params['height']
    f = params['format']
    c = params['codec']
    g = gpu_id
    if nvc.CudaVideoCodec.H264 == c:
        codec_name = 'h264'
    elif nvc.CudaVideoCodec.HEVC == c:
        codec_name = 'hevc'
    bsf_name = codec_name + '_mp4toannexb,dump_extra=all'

    cmd = [
        '/home/admin2/PycharmProjects/pythonProject/install/FFmpeg/build_x64_release_shared/bin/ffmpeg', '-hide_banner',
        '-rtsp_transport', 'tcp',
        # '-vsync', '2',
        '-i', url,
        '-c:v', 'copy',
        '-bsf:v', bsf_name,
        '-f', codec_name,
        '-bufsize', '300K',
        '-max_delay', '300',
        # '-vf',  'select=\'eq(pict_type\,I)\'',
        # '-s',           str(640)+'x'+str(640),
        'pipe:1',
        '-loglevel', 'quiet',
    ]
    # Run ffmpeg in subprocess and redirect it's output to pipe
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    print(url, proc.returncode, sep='\t')
    # w = 640
    # h = 640
    # Create HW decoder class
    nvdec = nvc.PyNvDecoder(w, h, f, c, g)
    # Amount of bytes we read from pipe first time.
    read_size = 4096
    # Total bytes read and total frames decded to get average data rate
    rt = 0
    fd = 0

    to_rgb = nvc.PySurfaceConverter(w, h,
                                    nvc.PixelFormat.NV12, nvc.PixelFormat.RGB,
                                    g)
    to_pln = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.RGB,
                                    nvc.PixelFormat.RGB_PLANAR, g)
    cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601,
                                             nvc.ColorRange.JPEG)

    # Main decoding loop, will not flush intentionally because don't know the
    # amount of frames available via RTSP.
    count = 0
    while True:
        if proc.returncode is not None:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            continue
        # Pipe read underflow protection
        if not read_size:
            read_size = int(rt / fd)
            # Counter overflow protection
            rt = read_size
            fd = 1

        # Read data.
        # Amount doesn't really matter, will be updated later on during decode.
        bits = proc.stdout.read(read_size)
        if not len(bits):
            print("Can't read data from pipe")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            continue
        else:
            rt += len(bits)

        # Decode
        enc_packet = np.frombuffer(buffer=bits, dtype=np.uint8)
        pkt_data = nvc.PacketData()
        try:
            surf = nvdec.DecodeSurfaceFromPacket(enc_packet, pkt_data)
            if not surf.Empty():
                fd += 1
                # Shifts towards underflow to avoid increasing vRAM consumption.
                if pkt_data.bsl < read_size:
                    read_size = pkt_data.bsl
                # Print process ID every second or so.
                # if not fd % fps:
                #     print(name)
                if not fd % 8:
                    surf = to_rgb.Execute(surf, cc_ctx)
                    surf = to_pln.Execute(surf, cc_ctx)
                    surf = surf.PlanePtr()
                    if surf.ElemSize() == 1:
                        img_tensor = pnvc.makefromDevicePtrUint8(surf.GpuMem(),
                                                                 surf.Width(),
                                                                 surf.Height(),
                                                                 surf.Pitch(),
                                                                 surf.ElemSize())
                        img_tensor.resize_(3, int(surf.Height() / 3), surf.Width())
                        torch.cuda.empty_cache()
                        # 转为torch.Tensor
                        img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
                        torch.cuda.empty_cache()
                        # print(img_tensor.shape)
                        # img_tensor.share_memory_()
                        # img_tensor = img_tensor.detach()
                        qu.put((url, img_tensor, time.time()))
                        # del img_tensor
                        torch.cuda.empty_cache()
                        count += 1
                        # img = img_tensor.permute(1, 2, 0).cpu().detach().numpy()
                        # cv2.imwrite('/ai35liyx/' + url.split('.')[-1] + '/' + str(count) + '_' + str(fd) + '.jpg', img.astype(np.uint8))
                        # count += 1
                        # time.sleep(10)

        # Handle HW exceptions in simplest possible way by decoder respawn
        except nvc.HwResetException:
            nvdec = nvc.PyNvDecoder(w, h, f, c, g)
            continue


def get_test(qu):
    retinaface = Retinaface()
    while True:
        dict_img = {}
        url, img_tensor, t1 = qu.get()
        dict_img[url] = img_tensor
        print('get img_tensor +', str(url), '+', img_tensor.device, '+', t1)
        del img_tensor
        torch.cuda.empty_cache()
        retinaface.detect_image(dict_img, 'output/result', t1, url)
        # print('finish')
        # print('get img_tensor +', str(name), '+, img_tensor.device)


if __name__ == "__main__":
    # 使用生产者-消费者模式，生产者解码视频流，消费者处理图像帧
    # idconnect = '172.18.4.228'

    qu = Queue(maxsize=20)
    gpuID = 0
    # 摄像头rtsp流地址
    urls = [
        # 'rtsp://admin:huawei123@192.168.31.24',
        # 'rtsp://admin:huawei123@192.168.29.36',
        # 'rtsp://admin:huawei123@192.168.29.30',
        # 'rtsp://admin:huawei123@192.168.29.40',
        # 'rtsp://admin:huawei123@192.168.29.26',
        # 'rtsp://admin:huawei123@192.168.29.29',
        # 'rtsp://admin:huawei123@192.168.29.27',
        "rtsp://admin:huawei123@192.168.30.17",
        "rtsp://admin:huawei123@192.168.30.60",
    ]
    pool_put, pool_get = [], []

    # 生产者：解码视频流，获取图像帧，放入队列中
    for i, url in enumerate(urls):
        # print(url)
        client = Process(target=rtsp_client, args=(
            url, i, gpuID, qu))
        client.start()
        pool_put.append(client)

    # 消费者：从队列中取出图像帧，并进行处理
    for _ in range(1):
        consumer = Process(target=get_test, args=(qu,))
        consumer.start()
        pool_get.append(consumer)

    for client in pool_get:
        client.join()
    for client in pool_put:
        client.join()
