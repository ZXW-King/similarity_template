#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: convert_onnx.py
@time: 2023/3/9 下午6:32
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
import torchvision.models as models
import torch
import cv2
import numpy as np
from onnxmodel import ONNXModel

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="",help="model path")
    parser.add_argument("--output", type=str, default="model.onnx",help="output model path")
    parser.add_argument("--image", type=str, default="",help="test image file")

    args = parser.parse_args()
    return args

def main():
    args = GetArgs()
    model = models.resnet18(pretrained=True)
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    onnx_input = torch.rand(1, 3, 224, 224)
    onnx_input = onnx_input.to("cuda:0")
    torch.onnx.export(model,
                      onnx_input,
                      args.output,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])

    # test_onnx(args.image, args.output)


if __name__ == '__main__':
    main()
