import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ghost_model import ghostnet

# 加载预训练的ghost_V1模型
model = ghostnet()
model.load_state_dict(torch.load('./state_dict_73.98.pth'))
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    test_list = ["shuiping","bin","yizi"]
    for t in test_list:
        # 加载并预处理待匹配的图像
        image1 = Image.open(f'/media/xin/work1/github_pro/similarity/tempalte_match/template_data/my_test_data/{t}.jpg').convert('RGB')
        image1 = preprocess(image1)
        image1 = image1.unsqueeze(0)

        image2 = Image.open(f'/media/xin/work1/github_pro/similarity/tempalte_match/template_data/test_data/{t}.png').convert('RGB')
        image2 = preprocess(image2)
        image2 = image2.unsqueeze(0)

        # 使用MobileNetV2提取图像的特征向量
        with torch.no_grad():
            output1 = model(image1)
            output2 = model(image2)

        # 计算特征向量之间的相似度（余弦相似度）
        similarity = torch.nn.functional.cosine_similarity(output1, output2)
        print(f"{t}_cosine:",similarity.item())
