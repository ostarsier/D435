import time

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import requests
import sys
from b2_utils import encode_image_to_base64, base64_to_pyobj, draw_detections, euler_to_quat, get_best_result, \
    get_obj_depth, get_real_coord

import math
# 配置并启动管道
pipeline = rs.pipeline()

config = rs.config()

# 启用流：彩色图像 (640x480 @ 30fps)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启用深度流 (640x480 @ 30fps)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 开始流传输
pipeline.start(config)

# 创建对齐对象：将深度帧对齐到彩色帧
align_to = rs.stream.color
align = rs.align(align_to)
time.sleep(3)

def object_detection(color_image, text_labels, threshold):#TODO
    bgr_image = color_image
    web_request_url = f"http://192.168.110.91:{8001}/obj_det"
    #score = 0.4
    encoded_image = encode_image_to_base64(bgr_image)
    input_dict = {
        "image_base64": encoded_image,
        "text_labels": text_labels,
        "score_threshold": threshold
    }
    input_json = json.dumps(input_dict)
    response = requests.post(web_request_url, json=input_json)
    output_dict = response.json()
    output_pickle_str = output_dict["output_pickle_str"]

    result = base64_to_pyobj(output_pickle_str)
    return result






try:
    while True:
        # 等待一组帧（颜色 + 深度）
        frames = pipeline.wait_for_frames()

        # 对齐帧
        aligned_frames = align.process(frames)

        # 获取对齐后的帧
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # test
        # x, y = 320, 240  # 待测像素
        # depth = aligned_depth_frame.get_distance(x, y)

        if not color_frame or not aligned_depth_frame:
            continue

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        text_labels = [["a solid red block", "a solid green block"]]
        text_labels = [["blue block"]]
        threshold = 0.4
        result = object_detection(color_image, text_labels, threshold=threshold)

        # 保存深度图识别结果
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        image_with_boxes = draw_detections(
            color_image,  # color_image
            result["boxes"],
            result["scores"],
            result["text_labels"],
            score_threshold=threshold
        )
        depth_filename = f'det_image.png'
        cv2.imwrite(depth_filename, image_with_boxes)
        print(f'Saved depth image to {depth_filename}')

        image_with_boxes = draw_detections(
            depth_colormap,  # color_image
            result["boxes"],
            result["scores"],
            result["text_labels"],
            score_threshold=threshold
        )
        depth_filename = f'det_image_depth.png'
        cv2.imwrite(depth_filename, image_with_boxes)
        print(f'Saved depth image to {depth_filename}')

        rt = get_best_result(result)
        if rt is None:
            print("未检测到目标")
            continue
        box, score, label = rt
        depth = get_obj_depth(box, depth_image)
        if depth is None:
            continue
        ##坐标计算
        # profile = pipeline.get_active_profile()
        # depth_stream = profile.get_stream(rs.stream.depth)

        # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # intr = depth_stream.as_video_stream_profile().get_intrinsics()
        # 对齐到谁，就拿谁的内参：
        color_intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()

        K = np.array([[color_intr.fx, 0, color_intr.ppx],
                      [0, color_intr.fy, color_intr.ppy],
                      [0, 0, 1]])
        """
        K
        [[607.70751953125, 0.0, 313.29962158203125], [0.0, 607.5650634765625, 258.92401123046875], [0.0, 0.0, 1.0]]
        """
        p_cam = get_real_coord(box, depth, K)
        #相机坐标系转机器人坐标系
        R = np.array([[0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0]])
        p_robot = R @ p_cam
        a = 0.075
        b = 0
        c = 0.4
        p_robot = p_robot + np.array([a, b, c])




        #move test
        """
                [0.395, -0.023011268816513136, 0.3757321194179909]
                """
        pass
        move(p_robot[0], p_robot[1], p_robot[2], 0.5, 0.5, 0.5, 0.5)


        roll = 0
        pitch = 0
        yaw = 0
        quat = euler_to_quat(roll, pitch, yaw)
        move(p_robot[0],p_robot[1],p_robot[2],quat[0],quat[1],quat[2],quat[3])

        roll = math.pi / 2
        pitch = 0
        yaw = 0
        quat = euler_to_quat(roll, pitch, yaw)
        move(p_robot[0], p_robot[1], p_robot[2], quat[0], quat[1], quat[2], quat[3])

        roll = 0
        pitch = math.pi / 2
        yaw = 0
        move(p_robot[0], p_robot[1], p_robot[2], quat[0], quat[1], quat[2], quat[3])
        pass
        #break




finally:  # 57 250
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()