# -*- coding: utf-8 -*-
"""
@copywrite, Ruizhe Lin and Peter Kner, University of Georgia, 2019
"""
import 3dsim_recon_p36 as si
import tifffile as tf
import numpy as np
#data file
fns = r'C:/Users/rl74173/Desktop/Image_based_ao_correction/20190115-140835_AlphaTN4_actin_si3d_ao_stacks_/20190115-140835_AlphaTN4_actin_si3d_ao_stacks_.tif'
img = tf.imread(fns)
#1st angle parameters
x1 = np.array([3.155, 0.354])
y1 = np.array([3.155, 0.354])
z1 = np.array([1.10])
#2nd angle parameters
x2 = np.array([-1.092, 0.358])
y2 = np.array([-1.092, 0.357])
z2 = np.array([1.10])
#3rd angle parameters
x3 = np.array([1.026, 0.339])
y3 = np.array([1.026, 0.338])
z3 = np.array([1.08])
#reconstruction
p = si.si3D(img, 5, 3, 0.670, 1.2)
#wiener filter
p.mu = 0.012
p.fwhm = 0.00001
p.strength = 0.99
p.thre = 1.
p.minv = 0.0
p.eta = 0.1
p.n = 1.
p.axy = 0.8
p.az = 0.8
#1st angle
p.separate(0)
p.shift0()
p.shift1(x1[0], x1[1], z1[0])
p.shift2(y1[0], y1[1])
a0 = p.getoverlap1(x1[0], x1[1], z1[0]) 
b0 = p.getoverlap2(y1[0], y1[1])
p.recon1(-a0[1], 1., -b0[1], 1.) #p.recon1(-a0[1], a0[0], -b0[1], b0[0])  #
#2nd angle
p.separate(1)
p.shift0()
p.shift1(x2[0], x2[1], z2[0])
p.shift2(y2[0], y2[1])
a1 = p.getoverlap1(x2[0], x2[1], z2[0])
b1 = p.getoverlap2(y2[0], y2[1])
p.recon_add(-a1[1], 1., -b1[1], 1.)  #p.recon_add(-a1[1], a1[0], -b1[1], b1[0])   #
#3rd angle
p.separate(2)
p.shift0()
p.shift1(x3[0], x3[1], z3[0])
p.shift2(y3[0], y3[1])
a2 = p.getoverlap1(x3[0], x3[1], z3[0])
b2 = p.getoverlap2(y3[0], y3[1])
p.recon_add(-a2[1], 1., -b2[1], 1.)  #p.recon_add(-a2[1], a2[0], -b2[1], b2[0])   #
#apodization
S = (p.Snum/p.Sden)*p.apd
finalimage = np.fft.fftshift(np.fft.ifftn(S))
#save image and otf
tf.imsave('final_image.tif',np.abs(finalimage).astype(np.float32),photometric='minisblack')
tf.imsave('effective_OTF.tif',np.abs(np.fft.fftshift(S)).astype(np.float32),photometric='minisblack')
