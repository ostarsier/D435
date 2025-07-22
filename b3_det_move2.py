import time

import pyrealsense2 as rs
import pickle
import numpy as np
import cv2
import json
import requests
import sys
from b2_utils import encode_image_to_base64, base64_to_pyobj, draw_detections, euler_to_quat, object_detection, \
    get_realsense_data, get_best_result, get_obj_depth, get_real_coord, move, set_gripper_position

import math

web_request_url = f"http://192.168.110.91:{8002}/get_realsense_data"#
web_request_url_det = f"http://192.168.110.91:{8003}/obj_det"
moveit_url = "http://192.168.110.91:8080/move"
redis_host="192.168.110.24"
#移动到初始位置
# print("移动到初始位置")
P_zero = [-0.00013586,-0.29682,-0.17101]
quat_zero = [-0.6191, 0.33828, 0.62199, -0.33971]
# rt=move(moveit_url,P_zero[0], P_zero[1], P_zero[2], quat_zero[0], quat_zero[1], quat_zero[2], quat_zero[3])
# time.sleep(1)
"""
Response body:
{
  "status": "success",
  "message": "规划和执行成功"
}
"""
#TODO

#get_realsense_data
result1=get_realsense_data(web_request_url)
color_image,depth_image,K=result1


##############################
# 使用pickle保存result1
#pickle_file_path = 'result1.pkl'
# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(result1, f)

# 使用pickle加载result1
# with open(pickle_file_path, 'rb') as f:
#     loaded_result1 = pickle.load(f)

#color_image,depth_image,K=loaded_result1
#######################################


#远程目标检测
#text_labels = [["a solid red block", "a solid green block"]]
text_labels = [["a solid blue block"]]
threshold = 0.15
try_times=0
success=False
bbox_result = object_detection(web_request_url_det,color_image, text_labels, threshold=threshold)
#获取单个目标
rt = get_best_result(bbox_result)
if rt is None:
    print("未检测到目标")
if len(bbox_result['boxes']) != 0:
    #获取深度
    box, score, label = rt
    depth = get_obj_depth(box, depth_image)
    if depth is None:
        print("无法获取有效深度值")
    else:
        success=True

#################################
if not success:
    exit(1) #TODO http返回
else:
    #####绘制目标(可选)#######
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    image_with_boxes = draw_detections(
        color_image,  # color_image
        bbox_result["boxes"],
        bbox_result["scores"],
        bbox_result["text_labels"],
        score_threshold=threshold
    )
    depth_filename = f'det_image.png'
    cv2.imwrite(depth_filename, image_with_boxes)
    print(f'Saved depth image to {depth_filename}')

    image_with_boxes = draw_detections(
        depth_colormap,  # color_image
        bbox_result["boxes"],
        bbox_result["scores"],
        bbox_result["text_labels"],
        score_threshold=threshold
    )
    depth_filename = f'det_image_depth.png'
    cv2.imwrite(depth_filename, image_with_boxes)
    print(f'Saved depth image to {depth_filename}')
    ###########################################################
    def move2coord(p_cam,qx, qy, qz, qw,offset_a,offset_b,offset_c):
        #计算机械臂坐标
        R = np.array([[0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0]])
        p_robot = R @ p_cam
        a = 0.075
        b = 0
        c = 0.4
        p_robot = p_robot + np.array([a, b, c])
        p_robot = p_robot + np.array([offset_a, offset_b, offset_c])
        print(f"移动到 {p_robot} {qx},{qy},{qz},{qw}")
        rt=move(moveit_url,p_robot[0], p_robot[1], p_robot[2], qx, qy, qz, qw)
        return rt
    #计算X，Y，Z坐标
    p_cam = get_real_coord(box, depth, K)
    #  p_cam=[-0.031322927421590326, -0.006828252403523044, 0.321]
    #============# 运动逻辑#====================
    #打开夹爪
    print("打开夹爪")
    set_gripper_position(redis_host,0.7)
    time.sleep(1.0)
    ######移动到测试位置：x=0.3116&y=-0.05&z=0.40&qx=0.5&qy=0.5&qz=0.5&qw=0.5################
    # print("移动到测试位置")
    # P = [0.3116, -0.05, 0.40]
    # # P=[-0.073,-0.199,9999]
    # quat = [0.5, 0.5, 0.5, 0.5]
    # rt = move(moveit_url, P[0], P[1], P[2], quat[0], quat[1], quat[2], quat[3])
    ############################################
    #移动到目标
    print("移动到目标")
    #(0.0, 0.0, 0.7071067811865475, 0.7071067811865476)
    roll = 0
    pitch = 0
    yaw = math.pi / 2

    # roll = math.pi / 2
    # pitch = 0
    # yaw = math.pi / 2
    # "http://localhost:8080/move?x=0.5216&y=-0.05&z=0.40&qx=0.99787&qy=0.0019314&qz=-0.052852&qw=0.038305"
    #quat = euler_to_quat(roll, pitch, yaw)
    quat=[0.25646, 0.096694,0.70286,0.65641]
    rt=move2coord(p_cam,quat[0],quat[1],quat[2],quat[3],offset_a=0.0,offset_b=0.07,offset_c=0.07)
    ############################################

    time.sleep(1)
    #关闭夹爪
    set_gripper_position(redis_host, 0.25)
    time.sleep(3.0)
    # 移动到初始位置
    print("移动到初始位置")
    rt=move(moveit_url,P_zero[0], P_zero[1], P_zero[2], quat_zero[0], quat_zero[1], quat_zero[2], quat_zero[3])
    #####################
