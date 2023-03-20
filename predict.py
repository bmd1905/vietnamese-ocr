import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
from PIL import Image
import difflib
import re
import math
import json
import sys
import argparse

import torch

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg

from PaddleOCR import PaddleOCR, draw_ocr

from VietnameseOcrCorrection.tool.predictor import Corrector
import time
from VietnameseOcrCorrection.tool.utils import extract_phrases

from ultis import display_image_in_actual_size


# Specifying output path and font path.
FONT = './PaddleOCR/doc/fonts/latin.ttf'


def predict(recognitor, detector, img_path, padding=4):
    # Load image
    img = cv2.imread(img_path)

    # Text detection
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    result = result[:][:][0]

    # Filter Boxes
    boxes = []
    for line in result:
        boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
    boxes = boxes[::-1]

    # Add padding to boxes
    padding = 4
    for box in boxes:
        box[0][0] = box[0][0] - padding
        box[0][1] = box[0][1] - padding
        box[1][0] = box[1][0] + padding
        box[1][1] = box[1][1] + padding

    # Text recognizion
    texts = []
    for box in boxes:
        cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        try:
            cropped_image = Image.fromarray(cropped_image)
        except:
            continue

        rec_result = recognitor.predict(cropped_image)
        text = rec_result#[0]

        texts.append(text)
        print(text)

    return boxes, texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--output', required='./runs/predict', help='path to save output file')
    parser.add_argument('--use_gpu', required=False, help='is use GPU?')
    args = parser.parse_args()

    # Configure of VietOCR
    # Default weight
    config = Cfg.load_config_from_name('vgg_transformer')
    # Custom weight
    # config = Cfg.load_config_from_file('vi00_vi01_transformer.yml')
    # config['weights'] = './pretrain_ocr/vi00_vi01_transformer.pth'

    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    config['device'] = 'mps'

    recognitor = Predictor(config)

    # Config of PaddleOCR
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=True)
    

    # Predict
    boxes, texts = predict(recognitor, detector, args.img, padding=2)


if __name__ == "__main__":    
    main()
