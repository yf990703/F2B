import cv2
import torch
import numpy as np
from utils.utils_bbox import match_face_body, create_combined_boxes, non_max_suppression

def recognize_person_with_face(yolo_instance, face_image, video_frame):
    # 将人脸照片转换为适应模型输入的大小
    face_image = cv2.resize(face_image, tuple(yolo_instance.input_shape))
    face_image = np.transpose(yolo_instance.preprocess_input(face_image.astype('float32')), (2, 0, 1))
    face_image = np.expand_dims(face_image, 0)

    with torch.no_grad():
        face_tensor = torch.from_numpy(face_image)
        if yolo_instance.cuda:
            face_tensor = face_tensor.cuda()

        # 将人脸输入网络进行预测
        face_outputs = yolo_instance.net(face_tensor)
        face_outputs = yolo_instance.decode_outputs(face_outputs, yolo_instance.input_shape)
        face_results = non_max_suppression(face_outputs, yolo_instance.num_classes, yolo_instance.input_shape,
                                           yolo_instance.input_shape, False, conf_thres=yolo_instance.confidence,
                                           nms_thres=yolo_instance.nms_iou)

    # 获取人脸匹配的人体框
    if face_results[0] is not None:
        top_face_label = np.array(face_results[0][:, 6], dtype='int32')
        top_face_conf = face_results[0][:, 4] * face_results[0][:, 5]
        top_face_boxes = face_results[0][:, :4]

        matched_pairs = match_face_body([top_face_boxes[0]], yolo_instance.body_boxes)
        combined_boxes = create_combined_boxes(matched_pairs)

        # 获取匹配的人体框的截图
        for box in combined_boxes:
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(video_frame.shape[0], np.floor(bottom).astype('int32'))
            right = min(video_frame.shape[1], np.floor(right).astype('int32'))

            person_image = video_frame[top:bottom, left:right, :]

            # 返回截图
            return person_image

    # 如果没有匹配的人体框，返回空值
    return None
