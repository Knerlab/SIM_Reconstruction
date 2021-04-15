# -*- coding: utf-8 -*-
"""
An example of running the 3D structured illumination microscopy image reconstruction codes
@copyright, Ruizhe Lin and Peter Kner, University of Georgia, 2021

"""

import sim_3drecon_p36 as si
import numpy as np

fns = r'###'  # raw 3dsim data file
p = si.si3D(fns,5,3,0.510,1.2)

#set parameters 
p.mu = 0.02
p.strength = 1.0
p.sigma = 4
p.eta = 0.1
p.expn = 1.    
p.axy = 0.8
p.az = 0.8
p.zoa = 1.

#initiate angles and spacings estimation
x0 = np.array([3.147, 0.285]) 
x1 =  np.array([-1.103, 0.286])
x2 =  np.array([1.019, 0.271])
z0 = np.array([1.45])
z1 = np.array([1.45])
z2 = np.array([1.35])

#search 1st angle
p.separate(0)
p.shift0() 
x0_2 = p.mapoverlap2(x0[0], x0[1], nps=10, r_ang=0.02, r_sp=0.02)
x0_2 = p.mapoverlap2(x0_2[0], x0_2[1], nps=10, r_ang=0.005, r_sp=0.005)
x0_2 = p.mapoverlap2(x0_2[0], x0_2[1], nps=10, r_ang=0.005, r_sp=0.005)
z0 = p.mapoverlapz(x0_2[0], x0_2[1], z0[0], nps=50, r_spz=0.25)
x0_1 = p.mapoverlap1(x0_2[0], x0_2[1], z0[0], nps=10, r_ang=0.005, r_sp=0.005)
x0_1 = p.mapoverlap1(x0_1[0], x0_1[1], z0[0], nps=10, r_ang=0.005, r_sp=0.005)

#search 2nd angle
p.separate(1)
p.shift0()
x1_2 = p.mapoverlap2(x1[0], x1[1], nps=10, r_ang=0.02, r_sp=0.02)
x1_2 = p.mapoverlap2(x1_2[0], x1_2[1], nps=10, r_ang=0.005, r_sp=0.005)
x1_2 = p.mapoverlap2(x1_2[0], x1_2[1], nps=10, r_ang=0.005, r_sp=0.005)
z1 = p.mapoverlapz(x1_2[0], x1_2[1],  z1[0], nps=50, r_spz=0.25)
x1_1 = p.mapoverlap1(x1_2[0], x1_2[1], z1[0], nps=10, r_ang=0.005, r_sp=0.005)
x1_1 = p.mapoverlap1(x1_1[0], x1_1[1], z1[0], nps=10, r_ang=0.005, r_sp=0.005)

#search 3rd angle
p.separate(2)
p.shift0()
x2_2 = p.mapoverlap2(x2[0], x2[1], nps=10, r_ang=0.02, r_sp=0.02)
x2_2 = p.mapoverlap2(x2_2[0], x2_2[1], nps=10, r_ang=0.005, r_sp=0.005)
x2_2 = p.mapoverlap2(x2_2[0], x2_2[1], nps=10, r_ang=0.005, r_sp=0.005)
z2 = p.mapoverlapz(x2_2[0], x2_2[1],  z2[0], nps=50, r_spz=0.25)
x2_1 = p.mapoverlap1(x2_2[0], x2_2[1], z2[0], nps=10, r_ang=0.005, r_sp=0.005)
x2_1 = p.mapoverlap1(x2_1[0], x2_1[1], z2[0], nps=10, r_ang=0.005, r_sp=0.005)    

#display results
print(x0_1)
print(x0_2)
print(z0)
print(x1_1)
print(x1_2)
print(z1)
print(x2_1)
print(x2_2)
print(z2)

#save results
fn = fns.split('.')[0] + '.txt'
np.savetxt(fn, ('1st angle', x0_1, z0, x0_2, 
                '2nd angle', x1_1, z1, x1_2, 
                '3rd angle', x2_1, z2, x2_2), fmt='%s')

# End of Parameter computation
del p

# Reconstruction
fnd = [r'###', r'###']  # all the raw 3dsim data to be reconstructed
fnd.sort()

for i in range(len(fnd)):
    fns = fnd[i]
    p = si.si3D(fns,5,3,0.510,1.2)
    
    #set parameters 
    p.mu = 0.04
    p.strength = 1.0
    p.sigma = 4
    p.eta = 0.1
    p.expn = 1.    
    p.axy = 0.8
    p.az = 0.8
    p.zoa = 1.
    
    #set spacing and angle
    x0_1 = np.array([3.149, 0.284])
    x0_2 = np.array([3.149, 0.285])
    z0 = np.array([1.63])
    x1_1 = np.array([-1.102, 0.286])
    x1_2 = np.array([-1.102, 0.286])
    z1 = np.array([1.64])
    x2_1 = np.array([1.021, 0.271])
    x2_2 = np.array([1.021, 0.271])
    z2 = np.array([1.37])
    
    #1st angle
    p.separate(0)
    p.shift0()
    p.shift1(x0_1[0], x0_1[1], z0[0])
    p.shift2(x0_2[0],x0_2[1])
    a0 = p.getoverlap1(x0_1[0], x0_1[1], z0[0]) 
    b0 = p.getoverlap2(x0_2[0], x0_2[1])
    p.recon1(-a0[1], a0[0], -b0[1], b0[0])
    
    #2nd angle
    p.separate(1)
    p.shift0()
    p.shift1(x1_1[0], x1_1[1], z1[0])
    p.shift2(x1_2[0],x1_2[1])
    a1 = p.getoverlap1(x1_1[0], x1_1[1], z1[0]) 
    b1 = p.getoverlap2(x1_2[0], x1_2[1])
    p.recon_add(-a1[1], a1[0], -b1[1], b1[0])
    
    #3rd angle
    p.separate(2)
    p.shift0()
    p.shift1(x2_1[0], x2_1[1], z2[0])
    p.shift2(x2_2[0],x2_2[1])
    a2 = p.getoverlap1(x2_1[0], x2_1[1], z2[0]) 
    b2 = p.getoverlap2(x2_2[0], x2_2[1])
    p.recon_add(-a2[1], a2[0], -b2[1], b2[0])
    
    #Apod
    S = ( p.Snum/p.Sden ) * p.apd
    finalimage = np.fft.ifftn(S)
    
    #save image and data
    tf.imsave(fns.split('.')[0]+'final_image.tif',np.fft.fftshift(finalimage).real.astype(np.float32),photometric='minisblack')
    tf.imsave(fns.split('.')[0]+'effective_OTF.tif',np.abs(np.fft.fftshift(S)).astype(np.float32),photometric='minisblack')
    fn = fns.split('.')[0] + '.txt'
    np.savetxt(fn, ('1st angle', x0_1, z0, a0, x0_2, b0, 
                    '2nd angle', x1_1, z1, a1, x1_2, b1, 
                    '3rd angle', x2_1, z2, a2, x2_2, b2), fmt='%s')
    del p
