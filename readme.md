
curl -X GET "http://192.168.110.91:8080/move?x=0.002&y=-0.199&z=-0.006&qx=0.5&qy=0.5&qz=0.5&qw=0.5"

curl -X GET "http://192.168.110.91:8080/move?x=0.3116&y=-0.05&z=0.40&qx=0.5&qy=0.5&qz=0.5&qw=0.5"

redis:
ssh yons@192.168.110.91
redis-cli

set joint_position [0,-0.5,0,0,0,9,0]

set joint_position [0.0,-0.5,0.0,0.0,0.0,0.0,0.0,0.0]
set joint_position [null,null,null,null,null,null,null,0.5]



主程序：
b3_det_move2.py
启动步骤：
启动/home/yons/cjd/obj_det_service中 service.py(目标检测服务) 与service_cam.py(视频流服务)
启动infer代码
启动moveit 和 ros http节点
启动b3_det_move2.py 测试