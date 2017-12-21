#!/usr/bin/python
# -*- coding: utf-8 -*-


'''
 光流图可视化
'''

import cv2
import flowlib
import struct
import os
import numpy as np
from scipy.misc import imread, imsave, imshow
flow_path = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'
flow2img_path = './test_out'


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


def view_flow(flow_path, flow_name):
    flow = read_flo(flow_path)
    flow_img = flowlib.flow_to_image(flow)
    png_name = flow_name.split('.')[0] + '.png'
    png_path = os.path.join(flow2img_path, png_name)
    imsave(png_path, flow_img)


# if __name__ == '__main__':
#     for i in range(22872):
#         flowname = os.path.join(flow_path, '%05d_flow.flo' % (i + 1))
#         print i + 1
#         view_flow(flowname, '%05d_flow.flo' % (i + 1))

if __name__ == '__main__':
    flowname = os.path.join(flow_path, '00002_flow.flo')
    # flodata = read_flo(flowname)
    view_flow(flowname, '00002_flow.flo')
    # print flodata.shape
