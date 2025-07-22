
# 机器人视觉抓取项目

本项目实现了一个基于视觉的机器人抓取系统。系统通过 RealSense 相机捕捉场景，利用一个独立的目标检测服务识别特定物体，计算出物体的三维空间坐标，并控制一个机械臂移动到目标位置进行抓取。

## 系统架构

系统由以下几个核心组件构成：

1.  **主控制脚本 (`b3_det_move2.py`)**: 整个流程的编排者。它负责调用其他服务，处理数据，并按顺序执行抓取任务。
2.  **相机服务 (`service_cam.py`)**: 运行在机器人控制PC上，通过Web API提供RealSense相机的彩色图像、深度图像和内参矩阵。
3.  **目标检测服务 (`service.py`)**: 运行在带有GPU的服务器上，接收图像和文本标签，返回检测到的物体的边界框（bounding boxes）、置信度和标签。
4.  **机械臂运动控制服务 (MoveIt & ROS HTTP Bridge)**: 提供HTTP接口，用于控制机械臂移动到指定的笛卡尔坐标和姿态。
5.  **夹爪控制 (Redis)**: 通过Redis消息队列来控制夹爪的打开和闭合。

## 主要功能

- **实时数据采集**: 从RealSense D435相机获取彩色和深度数据。
- **远程目标检测**: 将图像发送到远程服务器进行高效的目标检测。
- **三维坐标计算**: 结合深度数据和相机内参，将被检测物体的2D像素坐标转换为机器人基座标系下的3D坐标。
- **运动规划与执行**: 控制机械臂精确移动到目标物体上方。
- **夹爪控制**: 控制夹爪完成抓取和释放动作。

## 依赖

### Python 库
- `pyrealsense2`
- `numpy`
- `opencv-python` (`cv2`)
- `requests`
- `redis`

### 外部服务
- **相机服务**: 运行在 `http://192.168.110.91:8002`
- **目标检测服务**: 运行在 `http://192.168.110.91:8003`
- **MoveIt 服务**: 运行在 `http://192.168.110.91:8080`
- **Redis 服务器**: 运行在 `192.168.110.24`

## 配置

在 `b3_det_move2.py` 脚本的开头，可以配置以下参数：

- `web_request_url`: 相机服务的URL。
- `web_request_url_det`: 目标检测服务的URL。
- `moveit_url`: MoveIt运动控制服务的URL。
- `redis_host`: Redis服务器的主机名或IP。
- `P_zero`, `quat_zero`: 机械臂的初始/回归位置和姿态。
- `text_labels`: 需要检测的目标物体的文本描述，例如 `[["a solid blue block"]]`。
- `threshold`: 目标检测的置信度阈值。

## 启动与运行

请按照以下步骤启动和运行整个系统：

1.  **启动依赖服务**:
    - 在机器人控制PC上，启动相机服务 (`service_cam.py`) 和目标检测服务 (`service.py`)。这些服务位于 `/home/yons/cjd/obj_det_service` 目录下。
    - 确保 MoveIt 已经启动，并且 `ros_http_bridge` 节点正在运行，以提供HTTP接口。
    - 确保 Redis 服务器正在运行。

2.  **运行主脚本**:
    在所有服务都正常运行后，执行主控制脚本：
    ```bash
    python b3_det_move2.py
    ```

## 脚本逻辑分解 (`b3_det_move2.py`)

1.  **初始化**: 定义各个服务的URL和机械臂的初始位置。
2.  **获取相机数据**: 调用相机服务 (`get_realsense_data`) 获取当前的彩色图、深度图和相机内参 `K`。
3.  **目标检测**:
    - 定义要检测的物体 (`text_labels`)。
    - 调用目标检测服务 (`object_detection`)，将图像和标签发送过去。
    - 从返回结果中筛选出置信度最高的物体 (`get_best_result`)。
4.  **计算物理坐标**:
    - 如果检测到目标，使用 `get_obj_depth` 获取物体中心点的深度值。
    - 使用 `get_real_coord` 将2D像素坐标和深度值转换为相机坐标系下的三维坐标 `p_cam`。
5.  **运动控制**:
    - **坐标变换**: 在 `move2coord` 函数中，将相机坐标 `p_cam` 变换为机器人基座标系下的坐标 `p_robot`。这里包含了一个固定的旋转矩阵 `R` 和一个平移向量 `[a, b, c]`，以及额外的偏移量。
    - **打开夹爪**: 通过Redis发布消息，控制夹爪张开 (`set_gripper_position`)。
    - **移动到目标上方**: 调用 `move2coord` 函数，将机械臂末端移动到计算出的目标位置上方（带有一定偏移量）。
    - **抓取**: 通过Redis发布消息，控制夹爪闭合。
    - **返回初始位置**: 抓取成功后，控制机械臂返回到预设的初始位置 `P_zero`。

## 辅助工具 (`b2_utils.py`)

该文件包含一系列辅助函数，被主脚本 `b3_det_move2.py` 调用，以简化代码和重用逻辑。主要函数包括：
- `encode_image_to_base64`, `base64_to_pyobj`: 用于图像的编码和解码，以便通过HTTP传输。
- `draw_detections`: 在图像上绘制检测框。
- `euler_to_quat`: 欧拉角到四元数的转换。
- `object_detection`, `get_realsense_data`, `move`, `set_gripper_position`: 对各个Web服务的API调用进行了封装。
- `get_best_result`, `get_obj_depth`, `get_real_coord`: 用于处理检测结果和计算坐标的辅助函数。