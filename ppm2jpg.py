#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
from scipy.misc import imsave
input_img_path = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'
output_img_path = '/home/swq/Downloads/tensorflow-deeplab-resnet/flow_origin_1'

if __name__ == '__main__':
    for i in range(22872):
        input_img = os.path.join(input_img_path, '%05d_img1.ppm' % (i + 1))
        img = cv2.imread(input_img)
        output_img = os.path.join(output_img_path, '%05d_img1.jpg' % (i + 1))
        imsave(output_img, img)
    pass
