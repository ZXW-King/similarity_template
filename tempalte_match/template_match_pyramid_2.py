import cv2
import numpy as np
import time


# 图片旋转函数
def ImageRotate(img, angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)  # 绕图片中心进行旋转
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_rotation = cv2.warpAffine(img, M, (width, height))
    return image_rotation


# 金字塔下采样
def ImagePyrDown(image, NumLevels):
    for i in range(NumLevels):
        image = cv2.pyrDown(image)  # pyrDown下采样
    return image


def tm_ccoeff_normed_match(template_image, match_image, start_angle, end_angle, step,isRotate=False):
    res = cv2.matchTemplate(match_image, template_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    location = max_indx
    temp = max_val
    angle = 0
    if isRotate:
        for k_angle in range(start_angle, end_angle, step):
            template_image = ImageRotate(template_image, k_angle)
            res = cv2.matchTemplate(match_image, template_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
            if max_val > temp:
                location = max_indx
                temp = max_val
                angle = k_angle
    location_x = location[0]
    location_y = location[1]
    return (-angle, location_x, location_y,temp)


def tm_sqdiff_normed_match(template_image, match_image, start_angle, end_angle, step,isRotate=False):
    # 使用matchTemplate对原始灰度图像和图像模板进行匹配
    res = cv2.matchTemplate(match_image, template_image, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    location = min_indx
    temp = min_val
    angle = 0  # 当前旋转角度记录为0
    if isRotate:
        tic = time.time()
        # 以步长为5进行第一次粗循环匹配
        for i in range(start_angle, end_angle, step):
            template_image = ImageRotate(template_image, i)
            res = cv2.matchTemplate(match_image, template_image, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
            if min_val < temp:
                location = min_indx
                temp = min_val
                angle = i
        toc = time.time()
        print('匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')
    location_x = location[0]
    location_y = location[1]
    return (-angle, location_x, location_y,1-temp)


def match_tempalte_image(template_image, match_image, start_angle, end_angle, step,is_tm_sqdiff_normed):
    if is_tm_sqdiff_normed:
        result = tm_ccoeff_normed_match(template_image, match_image, start_angle, end_angle, step)
    else:
        result = tm_sqdiff_normed_match(template_image, match_image, start_angle, end_angle, step)
    match_point = {'angle': result[0], 'point': (result[1], result[2]),'confidence':result[3]}
    return match_point


# 旋转匹配函数（输入参数分别为模板图像、待匹配图像）
def RatationMatch(modelpicture, searchpicture,is_tm_sqdiff_normed):
    searchtmp = ImagePyrDown(searchpicture, 1)
    modeltmp = ImagePyrDown(modelpicture, 1)
    match_point = match_tempalte_image(modeltmp, searchtmp, -180, 180, 5,is_tm_sqdiff_normed)
    return match_point



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


# 画图
def draw_result(src, temp, match_point,confidence):
    cv2.rectangle(src, match_point,
                  (match_point[0] + temp.shape[1], match_point[1] + temp.shape[0]),
                  (0, 255, 0), 2)
    add_text(src, "Confidence:" + str(round(confidence, 2)))
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
    res_image = draw_result(match_image, ModelImage_edge, match_points['point'],match_points['confidence'])
    return res_image


if __name__ == '__main__':
    match_image = cv2.imread(
        "/media/xin/work1/github_pro/similarity/tempalte_match/test_data/yaoshi/yaoshis/yaoshi_1.jpeg")
    template_image = cv2.imread("/media/xin/work1/github_pro/similarity/tempalte_match/test_data/yaoshi/yaoshi_0.png")
    res = get_realsense(match_image, template_image,False)
    cv2.imshow('result', res)
    cv2.waitKey()
