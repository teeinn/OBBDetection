# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import BboxToolkit as bt
import numpy as np
import mmcv
from math import sqrt
import math


NOT_DRAW_BOXES = 0
ONLY_DRAW_BOXES = -1
ONLY_DRAW_BOXES_WITH_SCORES = -2

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'LightBlue', 'LightGreen'
]
FONT = ImageFont.load_default()


img_path = '/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/inference_img/area3/JPEGImages_new'
txt_path = '/media/qisens/2tb1/python_projects/training_pr/OBBDetection/work_dirs/resize_1024_with_augmentation/area3_save_dir/Task1_flatroof.txt'
output_path = '/media/qisens/2tb1/python_projects/training_pr/OBBDetection/work_dirs/resize_1024_with_augmentation/area3_save_dir/area3_total'

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(txt_path) as file:
    lines = [line.rstrip() for line in file]
    previous_filename = None
    for line in lines:
        factors = line.split(' ')
        filename, confidence, x0, y0, x1, y1, x2, y2, x3, y3 = factors[0], float(factors[1]), float(factors[2]), \
                                                               float(factors[3]), float(factors[4]), float(factors[5]), \
                                                               float(factors[6]), float(factors[7]), float(factors[8]), float(factors[9])
        if confidence < 0.1:
            continue

        if previous_filename != filename:
            if previous_filename == None:
                previous_filename = filename
                img = cv2.imread(os.path.join(img_path, filename + '.png'))
            else:
                cv2.imwrite(os.path.join(output_path, previous_filename+'.png'), img)
                print(previous_filename)
                img = cv2.imread(os.path.join(img_path, filename + '.png'))


        obbs = bt.poly2obb(np.array([x0, y0, x1, y1, x2, y2, x3, y3]))
        x_c, y_c, w, h, theta = obbs[0], obbs[1], obbs[2], obbs[3], obbs[4]

        Rad2Deg = 180.0 / math.pi
        theta = math.atan2(y2-y1,x2-x1) * Rad2Deg

        color = (0, 255, 0)
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, color, 2)

        previous_filename = filename

