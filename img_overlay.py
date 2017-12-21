#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
from scipy.misc import imsave
input_img_a = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'
input_img_b = '/home/swq/Downloads/tensorflow-deeplab-resnet/flow_contours_output'


if __name__ == '__main__':
    input_a = '00002_img2.ppm'
    input_b = 'offset.jpg'
    img_a = os.path.join(input_img_a, input_a)
    # img_b = os.path.join(input_img_b, input_b)

    imga = cv2.imread(img_a)
    imgb = cv2.imread(input_b)
    mg_mix = cv2.addWeighted(imga, 0.6, imgb, 0.4, 0)
    imsave('mg_mix22.png', mg_mix)
    # cv2.imshow('img', mg_mix)
    # cv2.waitKey(0)
    pass
