#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: test_onnx.py
@time: 2023/3/16 下午12:30
@desc: 
'''
import sys, os

from model2onnx.onnxmodel import ONNXModel

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
import cv2
import numpy as np


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, help="")
    parser.add_argument("--model", type=str, help="")

    args = parser.parse_args()
    return args

def test_onnx(img_path, model_file):
    model = ONNXModel(model_file)
    img_org = cv2.imread(img_path)
    img_org = cv2.resize(img_org, (224, 224), cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = np.transpose(img,[2,0,1])
    img = np.expand_dims(img, axis=0).astype("float32")

    output = model.forward(img)
    semi = output[0][0]
    print(semi)


def main():
    args = GetArgs()
    test_onnx(args.image, args.model)


if __name__ == '__main__':
    main()
