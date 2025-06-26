import pyrealsense2 as rs
import numpy as np
import cv2

from metric_depth.service import get_depth

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

try:
    while True:
        # 等待一组帧（颜色 + 深度）
        frames = pipeline.wait_for_frames()

        # 对齐帧
        aligned_frames = align.process(frames)

        # 获取对齐后的帧
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not aligned_depth_frame:
            continue

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # 获取当前时间戳用于文件名
        timestamp = cv2.getTickCount()
        
        # 保存彩色图像
        cv2.imwrite(f'color.png', color_image)
        
        # 深度图是单通道，我们进行归一化以便保存
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        cv2.imwrite(f'depth_aligned.png', depth_colormap)
        
        # 获取并保存第二个深度图
        depth_image2 = get_depth(color_image)
        # depth_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image2, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imwrite(f'depth_estimated.png', depth_colormap2)
        
        print(f'Saved images with timestamp: {timestamp}')
        
        # 添加短暂延迟，避免保存过快
        key = cv2.waitKey(1000)  # 每1秒保存一次
        if key == ord('q'):
            break

finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()