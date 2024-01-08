import argparse
import glob
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 加载预训练的ghost_V1模型
from LCNet.torchlcnet import TorchLCNet
from ghost_net.ghost_model import ghostnet
from utils import file


def add_text(img,text):
    # 定义字体类型、大小和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 0, 255)

    # 获取文本框的大小
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness=2)

    # 计算文本框左下角的坐标
    x = int((img.shape[1] - text_width) / 2)
    y = text_height + 10

    # 在图像的最上面中间绘制文本
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness=2)
    return img


def get_model(model_mode):
    if model_mode == "ghostnet":
        model = ghostnet()
        model.load_state_dict(torch.load('/media/xin/work1/github_pro/similarity/ghost_net/state_dict_73.98.pth'))
    elif model_mode == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
    elif model_mode == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_mode == "LCNet":
        model = TorchLCNet(scale=1.0)
        model.load_state_dict(torch.load("/media/xin/work1/github_pro/similarity/LCNet/model/PPLCNet_x1_0_pretrained.pth.tar"))
    else:
        return "model wrong!"
    model.eval()
    return model

# 图像预处理
def image_deal(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(Image.fromarray(image))
    image = image.unsqueeze(0)
    return image

def drow_image(img1,img2,similarity,filename):
    # 显示并保存每一对图像及其相似度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # 绘制两张不同的图像拼接
    h = max(h1, h2)
    w = w1 + w2
    # 创建一张新的画布
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # 将图像1拷贝到左侧
    canvas[0:h1, 0:w1, :] = img1
    # 将图像2拷贝到右侧
    canvas[0:h2, w1:w1 + w2, :] = img2
    add_text(canvas, "Confidence:" + str(round(similarity, 2)))
    cv2.imwrite("flower.png",canvas)
    return canvas


def draw_image(img1,img2,similarity,save_image):
    img1 = cv2.resize(img1,(256,256))
    img2 = cv2.resize(img2,(256,256))
    canvas = cv2.hconcat([img1,img2])
    add_text(canvas, "Confidence:" + str(round(similarity, 2)))
    cv2.imwrite(save_image,canvas)
    # cv2.imshow("res",canvas)
    # cv2.waitKey()

def get_max_similarity(image1,images):
    max_similarity = 0
    image22 = None
    for im in images:
        image2 = cv2.imread(im)
        image = image2.copy()
        image2 = image_deal(image2)
        with torch.no_grad():
            output1 = model(image1)
            output2 = model(image2)
        # 计算特征向量之间的相似度（余弦相似度）
        similarity = torch.nn.functional.cosine_similarity(output1, output2)
        if similarity.item() > max_similarity:
            max_similarity = similarity.item()
            image22 = image
        # print(similarity.item())
    return max_similarity,image22

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='相似度计算')
    parser.add_argument("--image1", type=str, help="图像1的路径")
    parser.add_argument("--image2", type=str, help="图像2的路径")
    parser.add_argument("--model_mode", type=str, default="ghostnet", help="计算方法,可选(ghostnet,mobilenet,resnet18,LCNet)")
    args = parser.parse_args()
    image1_path = args.image1
    image2_dir = args.image2
    # 加载并预处理待匹配的图像
    image1 = cv2.imread(image1_path)
    image11 = image1.copy()
    image1 = image_deal(image1)
    search = file.Walk(image2_dir, ["png", "jpeg", "jpg"])
    listing = [glob.glob(s)[0] for s in search]
    image_name = os.path.join("/media/xin/work1/github_pro/similarity/similar_match/test_res",
                              os.path.basename(image2_dir))
    os.makedirs(image_name, exist_ok=True)
    for m in ("ghostnet","mobilenet","resnet18","LCNet"):
        model = get_model(m)
        max_similarity,image22 = get_max_similarity(image1,listing)
        save_image = os.path.join(image_name, m + ".png")
        draw_image(image11, image22, max_similarity,save_image)
        print(save_image + "*************匹配完成！")

"""
--image
/media/xin/work1/github_pro/similarity/model2onnx/test_data/flower.jpg
--model
/media/xin/work1/github_pro/similarity/model2onnx/model.onnx
"""