import requests
import torch
import cv2
import numpy as np

from transformers import Owlv2Processor, Owlv2ForObjectDetection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# cache_path = "./hf_models/owlv2"
model_path = '/home/yons/media/hf_models/owlv2-large-patch14-ensemble'

processor = Owlv2Processor.from_pretrained(model_path)
model = Owlv2ForObjectDetection.from_pretrained(model_path)
model.to(device)
model.eval()

def object_detection(image_cv, text_labels,threshold):#TODO 每类一个thresholds
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    inputs = processor(text=text_labels, images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([(image_rgb.shape[0], image_rgb.shape[1])]).to(device)
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=text_labels
    )
    # Retrieve predictions for the first image for the corresponding text queries
    result = results[0]
    return result


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
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().detach().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().detach().numpy()

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
if __name__ == '__main__':
    # 使用OpenCV读取图像
    image_cv = cv2.imread("./color.png")
    # 将BGR转换为RGB（因为OpenCV默认读取为BGR格式）
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    text_labels = [["a solid red block", "a solid green block"]]
    text_labels = [["a solid green block"]]
    result = object_detection(image_cv, text_labels,0.4)
    result = object_detection(image_cv, text_labels,0.4)
    boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]

    image_with_boxes = draw_detections(
        image_cv,
        result["boxes"],
        result["scores"],
        result["text_labels"]
    )
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, image_with_boxes)
    print(f"Detection result saved to {output_path}")

    # 显示结果图像（可选）
    # cv2.imshow('Detection Result', image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()