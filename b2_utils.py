import json
import sys

import cv2
import numpy as np
import base64
import time
import pickle

import redis
import requests


def decode_base64_to_image(base64_string):
    # 将 Base64 字符串解码为字节
    img_data = base64.b64decode(base64_string)
    # 将字节转换为 numpy 数组
    nparr = np.frombuffer(img_data, np.uint8)
    # 将 numpy 数组解码为图像
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


def encode_image_to_base64(image):
    # 将图像编码为 JPEG 格式
    retval, buffer = cv2.imencode('.jpg', image)
    # 将图像数据转换为 Base64 字符串
    b64b = base64.b64encode(buffer)
    data_str = b64b.decode('utf-8')
    return data_str

def pyobj_to_base64(data):
    pickle_bytes = pickle.dumps(data)
    b64_bytes = base64.b64encode(pickle_bytes)
    return b64_bytes.decode("ascii")

def base64_to_pyobj(b64_str):

    pickle_bytes = base64.b64decode(b64_str.encode("ascii"))
    return pickle.loads(pickle_bytes)
# ── euler_to_quat_math.py ─────────────────────────────────────────────
"""
Convert Euler angles (roll, pitch, yaw) to quaternion (qx, qy, qz, qw).

Parameters
----------
r : float
    Roll  angle  (rad). Rotation about X axis.
p : float
    Pitch angle  (rad). Rotation about Y axis.
y : float
    Yaw   angle  (rad). Rotation about Z axis.
degrees : bool, optional
    If True, inputs are in degrees and will be internally converted to radians.

Returns
-------
(qx, qy, qz, qw) : tuple[float, float, float, float]
    Quaternion components, where qw is the scalar part.
"""
import math
from typing import Tuple

def euler_to_quat(r: float, p: float, y: float, *, degrees: bool = False) -> Tuple[float, float, float, float]:
    if degrees:                  # 支持角度制输入
        r, p, y = map(math.radians, (r, p, y))

    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)   # cos/sin(roll/2)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)   # cos/sin(pitch/2)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)   # cos/sin(yaw/2)

    # Z-Y-X (yaw-pitch-roll) 组合公式
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw

# ── quat_to_euler_math.py ────────────────────────────────────────────
"""
Convert quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw).

Parameters
----------
qx, qy, qz, qw : float
    Quaternion components (qw is scalar part).
degrees : bool, optional
    If True, return angles in degrees.

Returns
-------
(r, p, y) : tuple[float, float, float]
    Roll, pitch, yaw (rad by default).
"""
import math
from typing import Tuple

def quat_to_euler(qx: float, qy: float, qz: float, qw: float,
                  *, degrees: bool = False) -> Tuple[float, float, float]:
    # 1) 归一化，避免数值误差
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = [q / norm for q in (qx, qy, qz, qw)]

    # 2) 计算中间项
    # yaw (Z)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # pitch (Y)
    sinp = 2.0 * (qw*qy - qz*qx)
    # 处理数值超出 [-1,1] 导致 NaN
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)   # gimbal lock
    else:
        pitch = math.asin(sinp)

    # roll (X)
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    if degrees:
        roll, pitch, yaw = map(math.degrees, (roll, pitch, yaw))
    return roll, pitch, yaw


def get_realsense_data(web_request_url):
    input_dict = {}
    input_json = json.dumps(input_dict)
    response = requests.post(web_request_url, json=input_json)
    output_dict = response.json()
    output_pickle_str = output_dict["output_pickle_str"]
    restored = base64_to_pyobj(output_pickle_str)
    # print("\n恢复后的字典：", restored)
    color_image = restored["color_image"]
    depth_image = restored["depth_image"]
    K = restored["K"]
    return color_image,depth_image,K
def object_detection(web_request_url_det,color_image, text_labels, threshold):#TODO
    bgr_image = color_image

    #score = 0.4
    encoded_image = encode_image_to_base64(bgr_image)
    input_dict = {
        "image_base64": encoded_image,
        "text_labels": text_labels,
        "score_threshold": threshold
    }
    input_json = json.dumps(input_dict)
    response = requests.post(web_request_url_det, json=input_json)
    output_dict = response.json()
    output_pickle_str = output_dict["output_pickle_str"]

    result = base64_to_pyobj(output_pickle_str)
    return result
def get_best_result(result):
    # 将 PyTorch 张量转换为 NumPy 数组
    boxes = result["boxes"]
    scores = result["scores"]
    text_labels = result["text_labels"]

    # 获取scores中分数最高的索引
    if len(scores) > 0:
        max_score_idx = np.argmax(scores)
        max_score = scores[max_score_idx]
        best_box = boxes[max_score_idx]
        best_label = text_labels[max_score_idx]
        print(f"最高分数: {max_score:.4f}, 标签: {best_label}, 位置: {best_box}")
        return best_box, max_score, best_label
    else:

        return None

# return z坐标值
def get_obj_depth(box, depth_image):
    # 获取边界框的坐标
    x1, y1, x2, y2 = map(int, box)

    # 计算边界框的中心点
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # 定义中心点附近的区域（3x3像素区域）
    region_size = 4  # 中心点周围1个像素，形成3x3的区域
    # 收集区域内的非零深度值
    depth_values = []
    for y in range(max(0, center_y - region_size), min(depth_image.shape[0], center_y + region_size + 1)):
        for x in range(max(0, center_x - region_size), min(depth_image.shape[1], center_x + region_size + 1)):
            depth_value = depth_image[y, x]
            if depth_value > 0 and depth_value < 60000:  # 只考虑非零深度值 与非 65535
                depth_values.append(depth_value)
    # 计算非零深度值的平均值
    if depth_values:
        avg_depth = sum(depth_values) / len(depth_values)
        print(f"物体深度: {avg_depth:.2f}毫米")
        return avg_depth
    else:
        return None

def draw_detections(image_cv, boxes, scores, text_labels, score_threshold=0.4):
    """
    在图像上绘制检测框和标签

    参数:
        image_cv: 输入的OpenCV格式图像(BGR)
        boxes: 检测框坐标列表，格式为[x1, y1, x2, y2]
        scores: 置信度分数列表
        text_labels: 标签文本列表
        score_threshold: 分数阈值，低于此值的检测结果将被忽略

    返回:
        绘制了检测框和标签的图像
    """
    # 创建图像副本，避免修改原始图像
    image_with_boxes = image_cv.copy()

    # 设置字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    box_thickness = 2

    # 确保boxes, scores, text_labels是列表或numpy数组
    # if torch.is_tensor(boxes):
    #     boxes = boxes.cpu().detach().numpy()
    # if torch.is_tensor(scores):
    #     scores = scores.cpu().detach().numpy()

    # 为每个检测结果绘制框和标签
    for box, score, text_label in zip(boxes, scores, text_labels):
        if score < score_threshold:
            continue

        box = [int(round(i)) for i in box]  # 转换为整数坐标
        x1, y1, x2, y2 = box

        # 随机生成颜色（基于标签名称的哈希值）
        color = (hash(text_label) % 180, hash(text_label + '1') % 180, hash(text_label + '2') % 180)

        # 绘制检测框
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, box_thickness)

        # 准备标签文本
        label = f"{text_label}: {score:.2f}"

        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # 绘制文本背景
        cv2.rectangle(image_with_boxes, (x1, y1 - text_height - 10),
                      (x1 + text_width + 5, y1), color, -1)

        # 绘制文本
        cv2.putText(image_with_boxes, label, (x1, y1 - 5),
                    font, font_scale, (255, 255, 255), font_thickness)

        print(f"Detected {text_label} with confidence {score:.3f} at location {box}")

    return image_with_boxes
def get_real_coord(box, depth, K):
    '''
    已知物体 像素坐标点 (x,y)与深度值depth 以及 相机内参，如何计算物体在相机坐标系下的物理坐标
    '''
    # 获取边界框的坐标
    x1, y1, x2, y2 = map(int, box)

    # 计算边界框的中心点
    u = (x1 + x2) // 2
    v = (y1 + y2) // 2
    depth = depth / 1000

    """
        将像素坐标 + 深度转换到相机坐标系 (米)
        u, v: int/float, 像素坐标
        depth: float, 物理深度值 (m)
        K: 3x3 numpy 数组，相机内参
        返回: np.array([X, Y, Z]) (m)
    """

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy  # 若希望Y朝上，可加负号
    Z = depth
    print(X, Y, Z)
    return np.array([X, Y, Z])

class ApiResponseError(Exception):
    """Custom exception for API response errors."""
    pass

def move(moveit_url,x, y, z, qx, qy, qz, qw):

    params = {
        "x": x,
        "y": y,
        "z": z,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }

    max_retries = 5  # 最大重试次数
    retry_delay = 0  # 重试间隔时间（秒）

    for attempt in range(max_retries):
        try:
            # print(f"Attempt {attempt + 1}/{max_retries} to call {moveit_url}") # Optional debug line
            resp = requests.get(moveit_url, params=params, timeout=15)
            resp.raise_for_status()  # Raises HTTPError for 4xx/5xx responses

            try:
                response_data = resp.json()
            # Using requests.exceptions.JSONDecodeError which is raised by resp.json()
            except requests.exceptions.JSONDecodeError as json_exc:
                error_message = f"Failed to decode JSON response. Content: {resp.text[:200]}" # Truncate for brevity
                raise ApiResponseError(error_message) from json_exc

            api_status = response_data.get("status")
            if api_status == "success":
                print("Status:", resp.status_code)
                print("Response body:")
                # Using json.dumps for potentially better formatting and handling non-ASCII
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
                return True  # Request successful and API status is success
            else:
                # API status is not "success" or status key is missing
                error_message = f"API status was '{api_status}'. Expected 'success'. Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}"
                raise ApiResponseError(error_message)

        except (requests.exceptions.RequestException, ApiResponseError) as exc:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {type(exc).__name__} - {str(exc)}")
            if attempt < max_retries - 1:
                if retry_delay > 0:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                elif retry_delay == 0:
                    print("Retrying immediately...")
            else:
                print(f"[ERROR] All retries failed after {max_retries} attempts. Giving up.", file=sys.stderr)
                return False # Return False from the function on ultimate failure
def set_gripper_position(host,position):
    # 连接到Redis服务器
    r = redis.Redis(host=host, port=6379, db=0)
    # 设置joint_position，使用null值
    joint_position = f"[null,null,null,null,null,null,null,{position},null,null,null,null,null,null,null,null]"
    for i in range(5):
        r.set('joint_position', joint_position)
        time.sleep(0.05)
    print(f"夹爪 joint_position设置成功: {joint_position}")

if __name__ == '__main__':
    roll, pitch, yaw = 0.0, 0.0, math.pi / 2
    q = euler_to_quat(roll, pitch, yaw)
    print("Quaternion:", q)

    rpy = quat_to_euler(*q)

    qx = 0.5
    qy = 0.5
    qz = 0.5
    qw = 0.5
    rpy2 = quat_to_euler(qx,qy,qz,qw)
    print("Euler (rad):", rpy)

    image_cv = cv2.imread("./color.png")
    # 编码图像
    encoded_image = encode_image_to_base64(image_cv)

    # 解码图像
    decoded_image = decode_base64_to_image(encoded_image)

    original = {"name": "Alice", "age": 30, "skills": ["python", "robotics"]}

    encoded = pyobj_to_base64(original)
    print("Base64 编码结果：\n", encoded)

    restored = base64_to_pyobj(encoded)
    print("\n恢复后的字典：", restored)