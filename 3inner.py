import pyrealsense2 as rs

pipeline = rs.pipeline()
profile = pipeline.start()

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

print("fx:", intr.fx)
print("fy:", intr.fy)
print("cx:", intr.ppx)
print("cy:", intr.ppy)

K = [[intr.fx, 0, intr.ppx],
     [0, intr.fy, intr.ppy],
     [0, 0, 1]]

pipeline.stop()