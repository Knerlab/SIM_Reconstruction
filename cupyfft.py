# -*- coding: utf-8 -*-
"""
@copywrite, Ruizhe Lin and Peter Kner, University of Georgia, 2019
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
