import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
import time
import matplotlib.pyplot as plt
from PIL import Image
import imagehash


def Histogram(image1, image2):
    # 加载要对比的两张图片
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # 将图片转为灰度图像
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    hist_img1 = cv2.calcHist([gray_img1],[0],None,[256],[0,256])
    hist_img2 = cv2.calcHist([gray_img2],[0],None,[256],[0,256])

    # 使用比较直方图相似性的函数计算相似度
    similarity = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

    #print(similarity)
    return similarity

def MSE(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    mse = np.mean((img1 - img2) ** 2)
    similarity = 1 / (mse + 1)
    #print(similarity)
    return similarity

def SSIM(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ssims = ssim(gray_img1, gray_img2)

    #print(ssims)
    return ssims
    # # 将图像转换为灰度
    # gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # # 计算SSIM
    # ssim_index, _ = cv2.compare_ssim(gray_image1, gray_image2, full=True)

    # # 打印结果
    # print(f"SSIM Index: {ssim_index}")

def Feature_ex(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建特征点检测器和描述符提取器
    orb = cv2.ORB_create()

    # 检测特征点并计算描述符
    kp1, desc1 = orb.detectAndCompute(gray_img1, None)
    kp2, desc2 = orb.detectAndCompute(gray_img2, None)

    # 使用 Brute-Force 方法匹配特征描述符
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # 根据匹配点的数量计算相似度
    similarity = len(matches)
    #print(similarity)
    return similarity

def calculate_image_similarity(image_path1, image_path2):
    # Load images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Calculate Phash for each image
    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)

    # Calculate Hamming distance between the hashes
    hamming_distance = hash1 - hash2

    # Normalize the distance (0 means identical images)
    similarity = 1.0 - (hamming_distance / len(hash1.hash) / 2.0)

    return similarity


def display_images_with_similarity(image1, image2, similarity,mode, elapsed_time):
    """
    Display two images side by side along with their similarity.

    Parameters:
    - image1: First image to display
    - image2: Second image to display (most similar image)
    - similarity: Similarity value between the images (0 to 1)

    Returns:
    None
    """
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Display Image 1
    ax1.imshow(image1)
    ax1.set_title('Image 1')
    ax1.axis('off')

    # Display Image 2 with the highest similarity
    ax2.imshow(image2)
    ax2.set_title('Image 2')
    ax2.axis('off')

    plt.suptitle(f'{mode} Similarity: {similarity:.5f}\nElapsed Time: {elapsed_time:.2f} seconds')
    plt.show()

def process_images_and_measure_time(image1 ,image2 ,mode):

    time_list = []
    start_time = time.time()
    for i in range(10):
        start_time = time.time()
        similarity = mode(image1, image2)
        end_time = time.time()
        time_list.append(end_time - start_time)
    new_list = time_list[1:-1]
    total_elapsed_time = sum(new_list)

    return total_elapsed_time

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='相似度计算')
    parser.add_argument("--image1", type=str, help="图像1的路径")
    parser.add_argument("--image2", type=str, help="图像2的路径")
    parser.add_argument("--mode", type=str, default="Histogram",help="计算方法,可选(Histogram,MSE,SSIM,FE,imagehash)")
    args = parser.parse_args()
    image1 = args.image1
    image2 = args.image2
    mode = args.mode
    start_time = time.time()
    if mode == "Histogram" :
        elapsed_time = process_images_and_measure_time(image1, image2, Histogram)
        simility = Histogram(image1 , image2)

    elif mode == "MSE" :
        elapsed_time = process_images_and_measure_time(image1, image2, MSE)
        simility = MSE(image1, image2)

    elif mode == "SSIM" :
        elapsed_time = process_images_and_measure_time(image1, image2, SSIM)
        simility = SSIM(image1, image2)

    elif mode == "FE" :
        elapsed_time = process_images_and_measure_time(image1, image2, Feature_ex)
        simility = Feature_ex(image1, image2)

    elif mode == "imagehash":
        elapsed_time = process_images_and_measure_time(image1, image2, calculate_image_similarity)
        simility = calculate_image_similarity(image1, image2)

    else:
        print("计算模式选择错误！！！   请重新选择!")
    end_time = time.time()
    display_images_with_similarity(image1, image2, simility, mode, elapsed_time)
    #print(f"花费时间: {end_time - start_time} seconds")


