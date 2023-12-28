"""
原理：
    模板匹配是用来在一副大图中搜寻查找模版图像位置的方法。OpenCV 为 我们提供了函数 cv2.matchTemplate()。
    和 2D 卷积一样 它也是用模板图像在输入图像 大图 上滑动 并在每一个位置对模板图像和与其对应的 输入图像的子区域  比较。
    OpenCV 提供了几种不同的比较方法 细节 看 文档 。
    返回的结果是一个灰度图像每一个像素值表示此区域与模板的匹配程度。
    如果输入图像的大小是 WxH，模板的大小是 wxh   输出的结果的大小就是 W-w+1 H-h+1 。
    当你得到这幅图之后 就可以使用函数 cv2.minMaxLoc() 来找到其中的最小值和最大值的位置了。
    第一个值为矩形左上角的点 位置；w h为模板矩形的宽和 。
    这个矩形就是找到的模板区域了。
方法：
    平方差匹配 CV_TM_SQDIFF：用两者的平方差来匹配，最好的匹配值为0
    归一化平方差匹配 CV_TM_SQDIFF_NORMED
    相关匹配 CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好
    归一化相关匹配CV_TM_CCORR_NORMED
    相关系数匹配 CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配
    归一化相关系数匹配 CV_TM_CCOEFF_NORMED
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/yaoshi_1.jpeg', 0)
img2 = img.copy()
template = cv2.imread('test_data/yaoshi/yaoshi_0.png', 0)

w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for i in range(len(methods)):
    img = img2.copy()
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
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('method: ' + methods[i])
    plt.show()
