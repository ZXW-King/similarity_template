import time

import cv2
import numpy as np


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



# 金字塔缩放参数，需要根据实际情况进行调整
scale_factor = 1 / 1.2

# 图像缩放
def processed_image(image, scale):
    height, width = image.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
    return img_resized


# 图像金字塔
def pyramid_image(image, current_scale=1.0):
    min_h = 12
    min_w = 12
    dim_h = 0
    dim_w = 0
    ratios = []
    images = []
    if len(image.shape) < 3:
        dim_h, dim_w = image.shape
    else:
        dim_h, dim_w, dim_c = image.shape
    while (dim_h > min_h and dim_w > min_w):
        current_scale *= scale_factor
        resized_img = processed_image(image, current_scale)
        images.append(resized_img)
        ratios.append(current_scale)
        dim_h, dim_w = resized_img.shape
    return images, ratios



def image_many_match(gray_src, tpl, methods, threshold=0.5):
    box_rec = []
    h, w = tpl.shape[:2]

    # 模板不能大于图像
    if h > min(gray_src.shape[:2]) or w > min(gray_src.shape[:2]):
        return box_rec
    res = cv2.matchTemplate(gray_src, tpl, methods)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        xmin = pt[0]
        ymin = pt[1]
        xmax = pt[0] + w
        ymax = pt[1] + h
        score = res[pt[1]][pt[0]]
        # 搜索框在图像左边，排除右边的干扰区域
        if xmax < gray_src.shape[1] / 2:
            box_rec.append([xmin, ymin, xmax, ymax, score])
    box_rec = np.array(box_rec)
    if len(box_rec) != 0:
        keep = py_cpu_nms(box_rec, 0.3)
        box_rec = box_rec[keep]
    return box_rec

def image_one_match(gray_src, tpl, methods, threshold=0.5):
    box_rec = []
    h, w = tpl.shape[:2]

    # 模板不能大于图像
    if h > min(gray_src.shape[:2]) or w > min(gray_src.shape[:2]):
        return box_rec
    res = cv2.matchTemplate(gray_src, tpl, methods)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)

    top_left = max_loc

    box_rec.append([top_left[0],top_left[1],top_left[0] + w, top_left[1] + h,max_val])
    box_rec = np.array(box_rec)
    if len(box_rec) != 0:
        keep = py_cpu_nms(box_rec, 0.3)
        box_rec = box_rec[keep]
    return box_rec

def pyramid_image_match(image,template,threshold=0.01):

    result_box = []
    # if image is None or template_white is None or template_black is None:
    #     return np.array(result_box)
    h_ratio = 180. / 1080.
    height, width, channel = image.shape
    roi_y =0
    roi_h = int(h_ratio * height)
    roi_img = image[roi_y:roi_y + roi_h, 0:width]
    gray_roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    gray_tem = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    current_scale = max(1.0, max(height * 1.0 / width, height / 1080.0))
    methods = cv2.TM_CCOEFF_NORMED
    if current_scale != 2339 / 1080.0:
        tmp_py_list, ratio_py_list = pyramid_image(gray_tem, current_scale)
        for tpl, tpl_ratio in zip(tmp_py_list, ratio_py_list):
            box_rec = image_one_match(gray_roi_img, tpl, methods, threshold)
            if len(box_rec) != 0:
                result_box.append(box_rec)
    else:
        box_rec = image_one_match(gray_roi_img, gray_tem, methods, threshold)
        if len(box_rec) != 0:
            result_box.append(box_rec)
    result_box = np.array(result_box)
    if result_box.size != 0:
        result_box = np.vstack(result_box)
        keep = py_cpu_nms(result_box, 0.3)
        result_box = result_box[keep]
        result_box.astype(int)

    return result_box


if __name__ == '__main__':
    image = cv2.imread("/media/xin/work1/github_pro/similarity/tempalte_match/test_data/yaoshi/yaoshis/yaoshi_1.jpeg")
    template = cv2.imread("/media/xin/work1/github_pro/similarity/tempalte_match/test_data/yaoshi/yaoshi_0.png")
    start = time.time()
    box_res = pyramid_image_match(image, template)
    if box_res.size != 0:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        for box in box_res:
            box = box[:4].astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                          (0, 0, 255), 2)
            cv2.imshow("image", image)
        cv2.waitKey(0)

