import time

import pyrealsense2 as rs
import numpy as np
import cv2

from a9_owlv2_api import object_detection, draw_detections

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
def get_best_result(result):
    # 将 PyTorch 张量转换为 NumPy 数组
    boxes = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    text_labels = result["text_labels"]
    
    # 获取scores中分数最高的索引
    if len(scores) > 0:
        max_score_idx = np.argmax(scores)
        max_score = scores[max_score_idx]
        best_box = boxes[max_score_idx]
        best_label = text_labels[max_score_idx]
        print(f"最高分数: {max_score:.4f}, 标签: {best_label}, 位置: {best_box}")
        return best_box,max_score,best_label
    else:

        return None

def get_obj_depth(box,depth_image):
    # 获取边界框的坐标
    x1, y1, x2, y2 = map(int, box)
    
    # 计算边界框的中心点
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # 定义中心点附近的区域（3x3像素区域）
    region_size = 2  # 中心点周围1个像素，形成3x3的区域
    
    # 收集区域内的非零深度值
    depth_values = []
    for y in range(max(0, center_y - region_size), min(depth_image.shape[0], center_y + region_size + 1)):
        for x in range(max(0, center_x - region_size), min(depth_image.shape[1], center_x + region_size + 1)):
            depth_value = depth_image[y, x]
            if depth_value > 0 and depth_value<60000:  # 只考虑非零深度值 与非 65535
                depth_values.append(depth_value)
    
    # 计算非零深度值的平均值
    if depth_values:
        avg_depth = sum(depth_values) / len(depth_values)
        print(f"物体深度: {avg_depth:.2f}毫米")
        return avg_depth
    else:
        print("无法获取有效深度值")
        return None

def get_real_coord(box,depth,K):
    '''
    已知物体 像素坐标点 (x,y)与深度值depth 以及 相机内参，如何计算物体在相机坐标系下的物理坐标
    '''
    # 获取边界框的坐标
    x1, y1, x2, y2 = map(int, box)

    # 计算边界框的中心点
    u = (x1 + x2) // 2
    v = (y1 + y2) // 2
    depth=depth/1000

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
    print(X,Y,Z)
    return np.array([X, Y, Z])


try:
    while True:
        # 等待一组帧（颜色 + 深度）
        frames = pipeline.wait_for_frames()

        # 对齐帧
        aligned_frames = align.process(frames)

        # 获取对齐后的帧
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        #test
        # x, y = 320, 240  # 待测像素
        # depth = aligned_depth_frame.get_distance(x, y)

        if not color_frame or not aligned_depth_frame:
            continue

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        text_labels = [["a solid red block", "a solid green block"]]
        text_labels = [["red block"]]
        threshold=0.4
        result = object_detection(color_image, text_labels,threshold=threshold)

        # 保存深度图识别结果
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        image_with_boxes = draw_detections(
            color_image,#color_image
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



        rt=get_best_result(result)
        if rt is None:
            print("未检测到目标")
            exit(0)
        box, score, label=rt
        depth=get_obj_depth(box,depth_image)

        ##坐标计算
        profile = pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs.stream.depth)

        #depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        #intr = depth_stream.as_video_stream_profile().get_intrinsics()
        #对齐到谁，就拿谁的内参：
        color_intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        print("fx:", color_intr.fx)
        print("fy:", color_intr.fy)
        print("cx:", color_intr.ppx)
        print("cy:", color_intr.ppy)

        K = np.array([[color_intr.fx, 0, color_intr.ppx],
             [0, color_intr.fy, color_intr.ppy],
             [0, 0, 1]])

        coord=get_real_coord(box,depth,K)

        pass


        

finally:#57 250
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()