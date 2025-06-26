import pyrealsense2 as rs
import numpy as np
import cv2


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

        # 显示图像
        cv2.imshow('Color Image', color_image)

        # 深度图是单通道，我们进行归一化以便显示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        cv2.imshow('Aligned Depth Image', depth_colormap)
        # 按下 q 键退出
        if cv2.waitKey(1) == ord('q'):
            break

finally:#57 250
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()