import math
import sys

import requests

from b2_utils import euler_to_quat

"""

curl -X GET "http://192.168.110.91:8080/move?x=0.0023&y=-0.199&z=-0.006&qx=0.5&qy=0.5&qz=0.5&qw=0.5"

curl -X GET "http://192.168.110.91:8080/move?x=0.3116&y=-0.05&z=0.40&qx=0.5&qy=0.5&qz=0.5&qw=0.5"


"""
def move(x, y, z, qx, qy, qz, qw):
    url = "http://192.168.110.91:8080/move"
    params = {
        "x": x,
        "y": y,
        "z": z,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)  # 10 s 超时
        resp.raise_for_status()  # 如果返回码不是 2xx 会抛出异常
        print("Status:", resp.status_code)
        print("Response body:")
        print(resp.text)
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] HTTP request failed: {exc}", file=sys.stderr)
        sys.exit(1)
#test1胸前位置
P=[0.3116,-0.05,0.40]

roll = 0
pitch = 0
yaw = math.pi/2
quat = euler_to_quat(roll, pitch, yaw)
#quat2=[0.5,0.5,0.5,0.5]
move(P[0],P[1],P[2],quat[0],quat[1],quat[2],quat[3])

###test2 返回初始位置
P=[0.00233,-0.199,-0.006]
quat=[0.5,0.5,0.5,0.5]
move(P[0],P[1],P[2],quat[0],quat[1],quat[2],quat[3])

#新初始位置
"""
目标位置: x=-0.073, y=-0.199, z=0.005
目标姿态(欧拉角): roll=3.141, pitch=1.271, yaw=-1.571
目标姿态(四元数): x=0.569, y=0.569, z=0.420, w=0.420
"""
P=[-0.073,-0.199,0.005]
quat=[0.569,0.569,0.420,0.420]
move(P[0],P[1],P[2],quat[0],quat[1],quat[2],quat[3])