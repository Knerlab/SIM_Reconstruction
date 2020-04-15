# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:44:40 2020

@author: linruizhe
"""

import cupy
cupy.fft.config.enable_nd_planning = True

def fftn(img):
    data = cupy.asarray(img)
    dataf = cupy.fft.fftn(data)
    imgf = cupy.asnumpy(dataf)
    return imgf

def ifftn(imgf):
    dataf = cupy.asarray(imgf)
    data = cupy.fft.ifftn(dataf)
    img = cupy.asnumpy(data)
    return img
