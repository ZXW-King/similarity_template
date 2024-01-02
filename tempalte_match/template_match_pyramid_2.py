import cv2
import numpy as np
import time


# 图片旋转函数
def ImageRotate(img, angle):  # img:输入图片；newIm：输出图片；angle：旋转角度(°)
    height, width = img.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    center = (width // 2, height // 2)  # 绕图片中心进行旋转
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_rotation = cv2.warpAffine(img, M, (width, height))
    return image_rotation


# 金字塔下采样
def ImagePyrDown(image, NumLevels):
    for i in range(NumLevels):
        image = cv2.pyrDown(image)  # pyrDown下采样
    return image


def tm_ccoeff_normed_match(template_image, match_image, start_angle, end_angle, step):
    res = cv2.matchTemplate(match_image, template_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    location = max_indx
    temp = max_val
    angle = 0
    for k_angle in range(start_angle, end_angle, step):
        newIm = ImageRotate(template_image, k_angle)
        res = cv2.matchTemplate(match_image, newIm, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
        if max_val > temp:
            location = max_indx
            temp = max_val
            angle = k_angle
    location_x = location[0]
    location_y = location[1]
    return (-angle, location_x, location_y)


def tm_sqdiff_normed_match(template_image, match_image, start_angle, end_angle, step):
    # 使用matchTemplate对原始灰度图像和图像模板进行匹配
    res = cv2.matchTemplate(match_image, template_image, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    location = min_indx
    temp = min_val
    angle = 0  # 当前旋转角度记录为0

    tic = time.time()
    # 以步长为5进行第一次粗循环匹配
    for i in range(start_angle, end_angle, step):
        newIm = ImageRotate(template_image, i)
        res = cv2.matchTemplate(match_image, newIm, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
        if min_val < temp:
            location = min_indx
            temp = min_val
            angle = i
    toc = time.time()
    print('匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')
    location_x = location[0]
    location_y = location[1]
    return (-angle, location_x, location_y)


def match_tempalte_image(template_image, match_image, start_angle, end_angle, step,is_tm_sqdiff_normed):
    if is_tm_sqdiff_normed:
        result = tm_ccoeff_normed_match(template_image, match_image, start_angle, end_angle, step)
    else:
        result = tm_sqdiff_normed_match(template_image, match_image, start_angle, end_angle, step)
    match_point = {'angle': result[0], 'point': (result[1], result[2])}
    return match_point


# 旋转匹配函数（输入参数分别为模板图像、待匹配图像）
def RatationMatch(modelpicture, searchpicture,is_tm_sqdiff_normed):
    searchtmp = ImagePyrDown(searchpicture, 3)
    modeltmp = ImagePyrDown(modelpicture, 3)
    match_point = match_tempalte_image(modeltmp, searchtmp, -180, 180, 5,is_tm_sqdiff_normed)
    return match_point


# 画图
def draw_result(src, temp, match_point):
    cv2.rectangle(src, match_point,
                  (match_point[0] + temp.shape[1], match_point[1] + temp.shape[0]),
                  (0, 255, 0), 2)
    return src


def get_realsense(match_image, template_image,is_tm_sqdiff_normed):
    ModelImage_edge = cv2.GaussianBlur(template_image, (5, 5), 0)
    SearchImage_edge = cv2.GaussianBlur(match_image, (5, 5), 0)
    tic = time.time()
    match_points = RatationMatch(ModelImage_edge, SearchImage_edge,is_tm_sqdiff_normed)
    toc = time.time()
    print('匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')
    print('匹配的最优区域的起点坐标为：' + str(match_points['point']))
    print('相对旋转角度为：' + str(match_points['angle']))
    res_image = draw_result(match_image, ModelImage_edge, match_points['point'])
    return res_image


if __name__ == '__main__':
    match_image = cv2.imread(
        "/media/xin/work1/github_pro/similarity/tempalte_match/test_data/yaoshi/yaoshis/yaoshi_1.jpeg")
    template_image = cv2.imread("/media/xin/work1/github_pro/similarity/tempalte_match/test_data/yaoshi/yaoshi_0.png")
    res = get_realsense(match_image, template_image,False)
    cv2.imshow('result', res)
    cv2.waitKey()
