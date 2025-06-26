import requests
import torch
import cv2
import numpy as np
from PIL import Image

from transformers import Owlv2Processor, Owlv2ForObjectDetection

# cache_path = "./hf_models/owlv2"
model_path = './hf_models/owlv2-large-patch14-ensemble'

processor = Owlv2Processor.from_pretrained(model_path)
model = Owlv2ForObjectDetection.from_pretrained(model_path)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# 使用OpenCV读取图像
image_cv = cv2.imread("./color.png")
# 将BGR转换为RGB（因为OpenCV默认读取为BGR格式）
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
# 转换为PIL图像以用于模型处理
image = Image.fromarray(image_rgb)
text_labels = [["a solid red block", "a solid green block"]]

inputs = processor(text=text_labels, images=image_rgb, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.tensor([(image.height, image.width)])
# Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
results = processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.4, text_labels=text_labels
)
# Retrieve predictions for the first image for the corresponding text queries
result = results[0]
boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
# 将PIL图像转换为OpenCV格式
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# 设置字体和颜色
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
box_thickness = 2

# 为每个检测结果绘制框和标签
for box, score, text_label in zip(boxes, scores, text_labels):
    box = [int(round(i)) for i in box.tolist()]  # 转换为整数坐标
    x1, y1, x2, y2 = box

    # 随机生成颜色（基于标签名称的哈希值）
    color = (hash(text_label) % 180, hash(text_label + '1') % 180, hash(text_label + '2') % 180)

    # 绘制检测框
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, box_thickness)

    # 准备标签文本
    label = f"{text_label}: {score:.2f}"

    # 计算文本大小
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # 绘制文本背景
    cv2.rectangle(image_cv, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)

    # 绘制文本
    cv2.putText(image_cv, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")

# 保存结果图像
output_path = "detection_result.jpg"
cv2.imwrite(output_path, image_cv)
print(f"Detection result saved to {output_path}")

# 显示结果图像（可选）
# cv2.imshow('Detection Result', image_cv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()