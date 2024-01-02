
import argparse
import glob
import os
import sys
# from tempalte_match.template_match_pyramid_2 import get_realsense
from template_match_pyramid_2 import get_realsense
import file
import cv2
import numpy as np
from matplotlib import pyplot as plt
from multi_scale_match import template_matching_with_rectangle


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

def tempalte_match_one_image(template, image, methods):
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    images = []
    for i in range(len(methods)):
        img1 = image.copy()
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        method = eval(methods[i])
        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # 使用不同的比较方法,对结果的解释不同
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            if "NORMED" in methods[i]:
                confidence = 1 - min_val
        else:
            top_left = max_loc
            if "NORMED" in methods[i]:
                confidence = max_val

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img1, top_left, bottom_right, 255, 2)
        images.append(img1)
        # 进行横向拼接
    result = cv2.hconcat([images[0], images[1]])
    add_text(result,"Confidence:" + str(round(confidence,2)))
    return result


def tempalte_match_many_image(template, image, method, threshold):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, eval(method))
    # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (255, 255, 0), 2)
    return image


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='template match')
    parser.add_argument('--input_template', type=str, default='',
                        help='Image path')
    parser.add_argument('--input_match_dir', type=str, default='',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--many_match', action='store_true',
                        help='Do not display images to screen.  (default: False).')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--pyramid_match', action='store_true',
                        help='Is it a pyramid match. (default: False).')
    parser.add_argument('--multi_scale', action='store_true',
                        help='Is it a multi_scale match. (default: False).')
    parser.add_argument('--write_dir', type=str, default='',
                        help='Directory where to write output frames (default: "").')
    opt = parser.parse_args()
    match_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                     'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for i in range(0,len(match_methods),2):
        methods = [match_methods[i], match_methods[i+1]]
        template = cv2.imread(opt.input_template)
        # shuiping:(77,167);miehuoqi:(91,236);yaoshi:(133,54);bin:(116,134)
        # template = cv2.resize(template,(116,134))
        search = file.Walk(opt.input_match_dir, ["png","jpeg","jpg"])
        listing = [glob.glob(s)[0] for s in search]
        for im_path in listing:
            directory, filename = os.path.split(im_path)
            match_image = cv2.imread(im_path)
            if match_image is None:
                raise Exception('Error reading image %s' % im_path)
            # 是否是金字塔匹配
            if opt.pyramid_match:
                if match_methods[i+1] == 'cv2.TM_SQDIFF_NORMED':
                    res = get_realsense(match_image, template, True)
                else:
                    res = get_realsense(match_image, template, False)
            #是否是多尺度匹配
            elif opt.multi_scale:
                res = template_matching_with_rectangle(template, match_image, False)
                
            else:
                if not opt.many_match:
                    res = tempalte_match_one_image(template, match_image, methods)
                else:
                    res = tempalte_match_many_image(template, match_image, match_methods[1], 0.8)
            print(f"{im_path}匹配成功！")
            if not opt.no_display:
                cv2.imshow("res", res)
                cv2.waitKey()
            if opt.write_dir:
                save_dir = os.path.join(opt.write_dir,methods[0].split(".")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".png")
                cv2.imwrite(save_path, res)
    print("测试完成！")