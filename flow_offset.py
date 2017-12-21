#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
import struct
from scipy.misc import imread, imsave, imshow
flow_path = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'
flow2img_path = './test_out'
flow_origin_path = './flow_out'
contours_img_path = '/home/swq/Downloads/tensorflow-deeplab-resnet/flow_contours_output'
UNKNOWN_FLOW_THRESH = 1e7


def over_zero(num):
    if num <= 0:
        num = 0
    return num


def max_flow(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    # print ("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))
    return maxu, maxv


def read_flo(floname):
    f = open(floname, "rb")
    data = f.read()
    f.close()
    width = struct.unpack('@i', data[4:8])[0]
    height = struct.unpack('@i', data[8:12])[0]
    flodata = np.zeros((height, width, 2))
    size = width * height
    for i in range(size):
        data_u = struct.unpack('@f', data[12 + 8 * i:16 + 8 * i])[0]
        data_v = struct.unpack('@f', data[16 + 8 * i:20 + 8 * i])[0]
        n = int(i / width)
        k = np.mod(i, width)
        flodata[n, k, :] = [data_u, data_v]
    return flodata


def offset(flow, origin_img, outname):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    rows = u.shape[0]
    cols = u.shape[1]
    new_img = np.ones([rows, cols, 3])
    maxu, maxv = max_flow(flow)

    for col in range(cols - abs(int(maxu))):
        for row in range(rows - abs(int(maxv))):
            new_img[row + int(v[row, col]), col + int(u[row, col]), :] = origin_img[row, col, :]
            new_img[row, col, :] = 0
    # print new_img.shape
    img = Image.fromarray(new_img.astype(np.uint8))
    img.save(outname)


if __name__ == '__main__':
    # flowname = os.path.join(flow_path, '00002_flow.flo')
    # flodata = read_flo(flowname)
    # originimg = cv2.imread('00002_flow.png')
    # offset(flodata, originimg)
    for i in range(122, 22872):
        print i
        flowname = os.path.join(flow_path, '%05d_flow.flo' % (i + 1))
        flodata = read_flo(flowname)
        originimg_path = os.path.join(contours_img_path, '%05d_flow.png' % (i + 1))
        originimg = cv2.imread(originimg_path)
        out_name = os.path.join(contours_img_path, '%05d_flow2.png' % (i + 1))
        offset(flodata, originimg, out_name)
    pass
