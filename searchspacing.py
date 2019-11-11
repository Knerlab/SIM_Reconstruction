# -*- coding: utf-8 -*-
"""
@copywrite, Ruizhe Lin and Peter Kner, University of Georgia, 2019
"""

import tifffile as tf
import 3dsim_recon_p36 as si

fns = r'' #raw data file
img = tf.imread(fns)

p = si.si3D(img,5,3,0.67,1.2)

p.mu = 0.04
p.fwhm = 0.00001
p.strength = 0.99
p.thre = 1.
p.minv = 0.0
p.eta = 0.1

spz = 1.05

p.separate(0)
p.shift0()
x1 = p.mapoverlap2(3.15, 0.425, nps=10, r_ang=0.02, r_sp=0.02)
x1 = p.mapoverlap2(x1[0], x1[1], nps=10, r_ang=0.005, r_sp=0.005)
x1 = p.mapoverlap2(x1[0], x1[1], nps=10, r_ang=0.005, r_sp=0.005)
y1 = p.mapoverlapz(x1[0], x1[1], spz, nps=40, r_spz=0.2)
z1 = p.mapoverlap1(x1[0], x1[1], y1[0], nps=8, r_ang=0.02, r_sp=0.02)
z1 = p.mapoverlap1(z1[0], z1[1], y1[0], nps=10, r_ang=0.005, r_sp=0.005)

p.separate(1)
p.shift0()
x2 = p.mapoverlap2(-1.095, 0.427, nps=10, r_ang=0.02, r_sp=0.02)
x2 = p.mapoverlap2(x2[0], x2[1], nps=10, r_ang=0.005, r_sp=0.005)
x2 = p.mapoverlap2(x2[0], x2[1], nps=10, r_ang=0.005, r_sp=0.005)
y2 = p.mapoverlapz(x2[0], x2[1], spz, nps=40, r_spz=0.2)
z2 = p.mapoverlap1(x2[0], x2[1], y2[0], nps=8, r_ang=0.02, r_sp=0.02)
z2 = p.mapoverlap1(z2[0], z2[1], y2[0], nps=10, r_ang=0.005, r_sp=0.005)

p.separate(2)
p.shift0()
x3 = p.mapoverlap2(1.022, 0.407, nps=10, r_ang=0.02, r_sp=0.02)
x3 = p.mapoverlap2(x3[0], x3[1], nps=10, r_ang=0.005, r_sp=0.005)
x3 = p.mapoverlap2(x3[0], x3[1], nps=10, r_ang=0.005, r_sp=0.005)
y3 = p.mapoverlapz(x3[0], x3[1], spz, nps=40, r_spz=0.2)
z3 = p.mapoverlap1(x3[0], x3[1], y3[0], nps=8, r_ang=0.02, r_sp=0.02)
z3 = p.mapoverlap1(z3[0], z3[1], y3[0], nps=10, r_ang=0.005, r_sp=0.005)
