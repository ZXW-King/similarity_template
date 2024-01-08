import cv2
import numpy as np

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


# 计算感知哈希值
def phash(image, hash_size=32):
    # 将图像缩放为64x64
    image = cv2.resize(image, (256, 256))
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算DCT系数
    dct = cv2.dct(np.float32(gray))
    # 保留左上角的8x8区域，抛弃低频分量
    dct_roi = dct[0:8, 0:8]
    # 计算平均值
    avg = np.mean(dct_roi)
    # 将DCT系数进行二值化
    hash = np.zeros((hash_size, hash_size), dtype=np.float32)
    for i in range(hash_size):
        for j in range(hash_size):
            hash[i, j] = 1 if dct[i, j] > avg else 0
    return hash

# 计算汉明距离
def hamming_distance(hash1, hash2):
    distance = np.sum(hash1 != hash2)
    return distance


# 计算相似度
def similarity(hash1, hash2):
    max_distance = len(hash1) * len(hash1[0])
    distance = hamming_distance(hash1, hash2)
    similarity = 1 - (distance / max_distance)
    return similarity

# 加载图像
# image1 = cv2.imread('/media/xin/work1/github_pro/similarity/tempalte_match/template_data/my_test_data/yizi.jpg')
# image2 = cv2.imread('/media/xin/work1/github_pro/similarity/tempalte_match/template_data/test_data/yizi.png')
# image1 = cv2.imread('/media/xin/work1/github_pro/similarity/tempalte_match/template_data/my_test_data/shuiping.jpg')
# image2 = cv2.imread('/media/xin/work1/github_pro/similarity/tempalte_match/template_data/test_data/shuiping.png')
image1 = cv2.imread('/media/xin/work1/github_pro/similarity/tempalte_match/template_data/my_test_data/bin.jpg')
image2 = cv2.imread('/media/xin/work1/github_pro/similarity/tempalte_match/template_data/test_data/bin.png')
bin1 = cv2.resize(image1,(256,256))
bin2 = cv2.resize(image1,(256,256))
res = cv2.hconcat([bin1,bin2])

# 计算哈希值
hash1 = phash(image1)
hash2 = phash(image1)

# 计算相似度
similarity_score = similarity(hash1, hash2)
add_text(res,"Confidence:" + str(round(similarity_score, 3)))
cv2.imwrite("/media/xin/work1/github_pro/similarity/tempalte_match/template_data/resize_img/res.png",res)
# 打印结果
print('相似度：', similarity_score)
