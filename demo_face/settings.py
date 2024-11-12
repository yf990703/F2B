
from sqlalchemy import create_engine, text
import os
import requests

engine = create_engine('mysql+pymysql://root:root@192.168.100.28:3306/position')

conn = engine.connect()

user_results = conn.execute(text('SELECT username, userphone, headurl FROM user')).fetchall()
# camera_results = conn.execute(text('SELECT stream_url, serverselection FROM camera')).fetchall()
# 编号，rtsp地址，分组
# 获取姓名,手机号,人脸图片

# camera_data = {}
user_data = {}

for row in user_results:
    name = row[0]
    phone = row[1]
    face_image = row[2]

    user_data[phone] = {
        'name': name,
        'face_image': face_image
    }

# for row in camera_results:
#     rtsp_url = row[0]
#     group = row[1]
#     parsed_url = urlparse(rtsp_url)
#     cam_ip = parsed_url.netloc.split('@')[-1]
#     # print(rtsp_url)
#
#     camera_data[cam_ip] = {
#         'rtsp_url': rtsp_url,
#         'group': group
#     }

conn.close()

print(user_data)
# print(camera_data)

if not os.path.exists('face_images'):
    os.makedirs('face_images')

for phone, data in user_data.items():
    name = data['name']
    face_image_url = data['face_image']

    filename = f'{name}.jpg'
    # while os.path.exists(os.path.join('face_images', filename)):
    #     filename = f'{phone}.jpg'

    response = requests.get(face_image_url)
    if response.status_code == 200:
        file_path = os.path.join('face_images', filename)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f'Successfully downloaded and saved {filename}')
    else:
        print(f'Failed to download {filename}')

cam_addres = [
    'rtsp://admin:huawei123@192.168.31.24',
    # 'rtsp://admin:huawei123@192.168.29.39',
    # 'rtsp://admin:huawei123@192.168.29.30',
    # 'rtsp://admin:huawei123@192.168.29.40',
    # 'rtsp://admin:huawei123@192.168.29.26',
    # 'rtsp://admin:huawei123@192.168.29.29',
    # 'rtsp://admin:huawei123@192.168.29.27',
    # "rtsp://admin:huawei123@192.168.30.17",
    # "rtsp://admin:huawei123@192.168.30.60",
    # "rtsp://admin:huawei123@192.168.31.115",
    "rtsp://admin:huawei123@192.168.31.118",
    # "rtsp://admin:huawei123@192.168.31.119",
    # "rtsp://admin:huawei123@192.168.31.117",
    # "rtsp://admin:huawei123@192.168.31.111",
    # "rtsp://admin:huawei123@192.168.31.116",
]

img_shape = [1080, 720]