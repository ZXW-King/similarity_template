import argparse
import glob
import os
import file
import cv2
import numpy as np
from matplotlib import pyplot as plt


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
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img1, top_left, bottom_right, 255, 2)
        images.append(img1)
        # 进行横向拼接
    result = cv2.hconcat([images[0], images[1]])
    return result


def tempalte_match_many_image(template, image, method, threshold):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
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
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--write_dir', type=str, default='',
                        help='Directory where to write output frames (default: "").')
    opt = parser.parse_args()
    match_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                     'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for i in range(0,len(match_methods),2):
        print(i)
        methods = [match_methods[i], match_methods[i+1]]
        template = cv2.imread(opt.input_template)
        search = file.Walk(opt.input_match_dir, ["png","jpeg","jpg"])
        listing = [glob.glob(s)[0] for s in search]
        for im_path in listing:
            directory, filename = os.path.split(im_path)
            grayim = cv2.imread(im_path)
            if grayim is None:
                raise Exception('Error reading image %s' % im_path)
            if not opt.many_match:
                res = tempalte_match_one_image(template, grayim, methods)
            else:
                res = tempalte_match_many_image(template, grayim, match_methods[1], 0.8)
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