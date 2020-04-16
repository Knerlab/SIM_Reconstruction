# -*- coding: utf-8 -*-
"""
@copywrite, Ruizhe Lin and Peter Kner, University of Georgia, 2019
"""

import sim3d_recon_p36 as si
import numpy as np
import tifffile as tf

fns = r'***' #image path

class sim3drecon(object):
    
    def __init__(self):
        self.p = si.si3D(fns,5,3,0.515,1.2)
        #set parameters 
        self.p.mu = 0.04
        self.p.strength = 1.0
        self.p.sigma = 4
        self.p.eta = 0.1
        self.p.expn = 1.    
        self.p.axy = 0.8
        self.p.az = 0.8
        self.p.zoa = 1.
        self.inter_results = False
        #angles and spacings estimation
        self.x0 = np.array([0., 0.280])        
        self.x1 = np.array([2.094, 0.280])
        self.x2 = np.array([4.189, 0.280])
        self.z0 = np.array([1.33])
        self.z1 = np.array([1.33])
        self.z2 = np.array([1.33])
        
    def compute(self):
        #search 1st angle
        self.p.separate(0)
        self.p.shift0() 
        self.x0_2 = self.p.mapoverlap2(self.x0[0], self.x0[1], nps=10, r_ang=0.02, r_sp=0.02)
        self.x0_2 = self.p.mapoverlap2(self.x0_2[0], self.x0_2[1], nps=10, r_ang=0.005, r_sp=0.005)
        # self.x0_2 = self.p.mapoverlap2(self.x0_2[0], self.x0_2[1], nps=10, r_ang=0.005, r_sp=0.005)
        self.z0 = self.p.mapoverlapz(self.x0_2[0], self.x0_2[1], self.z0[0], nps=40, r_spz=0.2)
        self.x0_1 = self.p.mapoverlap1(self.x0_2[0], self.x0_2[1], self.z0[0], nps=10, r_ang=0.005, r_sp=0.005)
        self.x0_1 = self.p.mapoverlap1(self.x0_1[0], self.x0_1[1], self.z0[0], nps=10, r_ang=0.005, r_sp=0.005)
        #search 2nd angle
        self.p.separate(1)
        self.p.shift0()
        self.x1_2 = self.p.mapoverlap2(self.x1[0], self.x1[1], nps=10, r_ang=0.02, r_sp=0.02)
        self.x1_2 = self.p.mapoverlap2(self.x1_2[0], self.x1_2[1], nps=10, r_ang=0.005, r_sp=0.005)
        # self.x1_2 = self.p.mapoverlap2(self.x1_2[0], self.x1_2[1], nps=10, r_ang=0.005, r_sp=0.005)
        self.z1 = self.p.mapoverlapz(self.x1_2[0], self.x1_2[1],  self.z1[0], nps=40, r_spz=0.2)
        self.x1_1 = self.p.mapoverlap1(self.x1_2[0], self.x1_2[1], self.z1[0], nps=10, r_ang=0.005, r_sp=0.005)
        self.x1_1 = self.p.mapoverlap1(self.x1_1[0], self.x1_1[1], self.z1[0], nps=10, r_ang=0.005, r_sp=0.005)
        #search 3rd angle
        self.p.separate(2)
        self.p.shift0()
        self.x2_2 = self.p.mapoverlap2(self.x2[0], self.x2[1], nps=10, r_ang=0.02, r_sp=0.02)
        self.x2_2 = self.p.mapoverlap2(self.x2_2[0], self.x2_2[1], nps=10, r_ang=0.005, r_sp=0.005)
        # self.x2_2 = self.p.mapoverlap2(self.x2_2[0], self.x2_2[1], nps=10, r_ang=0.005, r_sp=0.005)
        self.z2 = self.p.mapoverlapz(self.x2_2[0], self.x2_2[1],  self.z2[0], nps=40, r_spz=0.2)
        self.x2_1 = self.p.mapoverlap1(self.x2_2[0], self.x2_2[1], self.z2[0], nps=10, r_ang=0.005, r_sp=0.005)
        self.x2_1 = self.p.mapoverlap1(self.x2_1[0], self.x2_1[1], self.z2[0], nps=10, r_ang=0.005, r_sp=0.005)    
        
    def reconstruct(self):
        #1st angle
        self.p.separate(0)
        self.p.shift0()
        self.p.shift1(self.x0_1[0], self.x0_1[1], self.z0[0])
        self.p.shift2(self.x0_2[0],self.x0_2[1])
        a0 = self.p.getoverlap1(self.x0_1[0], self.x0_1[1], self.z0[0]) 
        b0 = self.p.getoverlap2(self.x0_2[0], self.x0_2[1])
        self.p.recon1(-a0[1], 1., -b0[1], 1.) #self.p.recon1(-a0[1], a0[0], -b0[1], b0[0])  #
        #2nd angle
        self.p.separate(1)
        self.p.shift0()
        self.p.shift1(self.x1_1[0], self.x1_1[1], self.z1[0])
        self.p.shift2(self.x1_2[0],self.x1_2[1])
        a1 = self.p.getoverlap1(self.x1_1[0], self.x1_1[1], self.z1[0]) 
        b1 = self.p.getoverlap2(self.x1_2[0], self.x1_2[1])
        self.p.recon_add(-a1[1], 1., -b1[1], 1.)  #self.p.recon_add(-a1[1], a1[0], -b1[1], b1[0])   #
        #3rd angle
        self.p.separate(2)
        self.p.shift0()
        self.p.shift1(self.x2_1[0], self.x2_1[1], self.z2[0])
        self.p.shift2(self.x2_2[0],self.x2_2[1])
        a2 = self.p.getoverlap1(self.x2_1[0], self.x2_1[1], self.z2[0]) 
        b2 = self.p.getoverlap2(self.x2_2[0], self.x2_2[1])
        self.p.recon_add(-a2[1], 1., -b2[1], 1.)  #self.p.recon_add(-a2[1], a2[0], -b2[1], b2[0])   #
        #Apod
        self.S = (self.p.Snum/self.p.Sden)*self.p.apd
        self.finalimage = np.fft.ifftn(S)
        
    def saveimg(self):
        tf.imsave('final_image.tif',np.fft.fftshift(self.finalimage).real.astype(np.float32),photometric='minisblack')
        tf.imsave('effective_OTF.tif',np.abs(np.fft.fftshift(self.S)).astype(np.float32),photometric='minisblack')
