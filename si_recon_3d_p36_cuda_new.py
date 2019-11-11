# -*- coding: utf-8 -*-
"""
3D Structured Illumination Reconstruction Algorothm with python 3.6 
Using CUDA to accelarate FFT computation   07/02/2018
Add Guassian filter, padding and backgroudn substraction(histogram)  04/29/2019
Add Apodization  06/22/2019

Note: Raw data z stack number must be even!!!

@copywrite, Ruizhe Lin and Peter Kner, University of Georgia, 2019
"""

import os
temppath = 'C:/Users/rl74173/Documents/Python Code/temp'
join = lambda fn: os.path.join(temppath,fn)

os.environ['NUMBAPRO_CUDALIB']=r'C:/Users/rl74173/AppData/Local/Continuum/anaconda3/pkgs/cudatoolkit-9.0-1/Library/bin'

import tifffile as tf
import psfsim36
from pylab import imshow, subplot, figure, plot
import numpy as np
from numba import cuda
from pyculib.fft.binding import Plan, CUFFT_C2C
from scipy.fftpack import fftshift
from scipy import signal

class si3D(object):

    def __init__(self,img_stack,nph,nangles,wavelength,na):
        self.img_stack = self.subback(img_stack)
        self.img_stack = np.pad(self.img_stack, ((2*nph*nangles,2*nph*nangles),(0,0),(0,0)),'constant', constant_values=(0))
        nz,nx,ny = self.img_stack.shape
        self.nz = int(nz/nph/nangles)
        self.nx = nx
        self.ny = ny
        self.mu = 1e-2
        self.wl = wavelength
        self.cutoff = 1e-3
        self.na = na
        self.dx = 0.089
        self.dz = 0.2
        self.nphases = nph
        self.norders = 5
        self.dpx = 1/((self.nx*2.)*(self.dx/2.))
        self.dpz = 1/((self.nz*2.)*(self.dz/2.))
        self.radius_xy = (2*self.na/self.wl)/self.dpx
        self.radius_z = ((self.na**2)/(2*self.wl))/self.dpz
        self.strength = 0.00001
        self.fwhm = 0.99
        self.minv = 0.0
        self.thre = 1.
        self.minv = 0.0
        self.sigma = 4.
        self.eta = 0.08
        self.n = 1.
        self.axy = 0.8
        self.az = 0.8
        self.psf = self.getpsf()
        self.sepmat = self.sepmatrix()
        self.meshgrid()
        self.winf = self.window(self.eta)
        self.apd = self.apod()
        self.img_stack = self.img_stack.reshape(self.nz,nangles,nph,nx,ny).swapaxes(0,1).swapaxes(1,2)

    def subback(self,img):
        nz, nx, ny = img.shape
        for i in range(nz):
            data = img[i,:,:]
            hist, bin_edges = np.histogram(data, bins=np.arange(data.min(),data.max()) )
            ind = np.where(hist == hist.max())
            bg = bin_edges[np.max(ind)+1]
            data[data<=bg] = 0.
            data[data>bg] = data[data>bg] - bg
            img[i,:,:] = data
        return img
        
    def meshgrid(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        x = np.arange(2*nx)
        y = np.arange(2*ny)
        z = np.arange(2*nz)
        zv, xv, yv = np.meshgrid(z, x, y, indexing='ij', sparse=True)
        i_zv = zv[1:nz+1,0,0]
        zv[nz:2*nz,0,0] = -np.flip(i_zv,0)
        i_xv = xv[0,1:nx+1,0]
        xv[0,nx:2*nx,0] = -np.flip(i_xv,0)
        i_yv = yv[0,0,1:ny+1]
        yv[0,0,ny:2*ny] = -np.flip(i_yv,0)
        self.zv = zv
        self.xv = xv
        self.yv = yv

    def shiftmat(self,kz,kx,ky):
        zv = self.zv
        xv = self.xv
        yv = self.yv
        a = np.exp(2j*np.pi*(kx*xv+ky*yv))*np.cos(2*np.pi*kz*zv)
        return a.astype(np.complex64)
    
    def sepmatrix(self):
        nphases = self.nphases
        norders = self.norders
        sepmat = np.zeros((norders,nphases),dtype=np.float32)
        norders = int((norders+1)/2)
        phi = 2*np.pi/nphases
        for j in range(nphases):
            sepmat[0, j] = 1.0/nphases
            for order in range(1,norders):
                sepmat[2*order-1,j] = 2.0 * np.cos(j*order*phi)/nphases
                sepmat[2*order  ,j] = 2.0 * np.sin(j*order*phi)/nphases
        return sepmat

    def getpsf(self):
        nz = self.nz
        nx = self.nx * 2
        dz = self.dz / 2
        dx = self.dx / 2
        wl = self.wl
        na = self.na
        lim1 = -dz*nz
        lim2 = dz*nz
        psf = psfsim36.psf()
        psf.setParams(wl=wl,na=na,dx=dx,nx=nx)
        psf.getFlatWF()
        psf.get3Dpsf(lim1,lim2,dz)
        psf1 = psf.stack
        psf1 = (psf1/psf1.sum()).astype(np.complex64)
        return psf1

    def separate(self, nangle=0):
        npx = self.nz*self.nx*self.ny
        nz = self.nz
        nx = self.nx
        ny = self.ny
        out = np.dot(self.sepmat,self.img_stack[nangle].reshape(self.nphases,npx))
        self.img_0 = fftshift(self.interp(out[0].reshape(nz,nx,ny))*self.winf).astype(np.complex64)
        self.img_1_0 = fftshift(self.interp((out[1]+1j*out[2]).reshape(nz,nx,ny))*self.winf).astype(np.complex64)
        self.img_1_1 = fftshift(self.interp((out[1]-1j*out[2]).reshape(nz,nx,ny))*self.winf).astype(np.complex64)
        self.img_2_0 = fftshift(self.interp((out[3]+1j*out[4]).reshape(nz,nx,ny))*self.winf).astype(np.complex64)
        self.img_2_1 = fftshift(self.interp((out[3]-1j*out[4]).reshape(nz,nx,ny))*self.winf).astype(np.complex64)

    def getoverlap1(self,angle,spacingx,spacingz):
        ''' shift 2nd order data '''
        dx = self.dx / 2
        dz = self.dz / 2
        kx = dx*np.cos(angle)/(spacingx*2)
        ky = dx*np.sin(angle)/(spacingx*2)
        kz = dz/spacingz
        
        nxh = self.nx
        nyh = self.ny
        
        ysh = self.shiftmat(kz,kx,ky).astype(np.complex64)
        otf = self.cufftn((self.psf*ysh))
        yshf = np.abs(self.cufftn(ysh))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<nxh):
            sx = sx
        else:
            sx = sx-2*nxh
        if (sy<nyh):
            sy = sy
        else:
            sy = sy-2*nyh 
        zsp = self.zerosuppression( sz, sx, sy)
        otf = otf * zsp
        
        ysh = self.shiftmat(0.,kx,ky).astype(np.complex64)
        imgf = self.img_1_0.astype(np.complex64)
        imgf = self.cufftn((imgf*ysh))
        
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf*imgf0
        wimgf1 = otf0*imgf
        msk = (np.abs(otf0*otf)>cutoff).astype(np.complex64)
        a = np.sum(msk*wimgf1*wimgf0.conj())/np.sum(msk*wimgf0*wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        return mag, phase

    def mapoverlap1(self,angle,spacing,spz,nps=10,r_ang=0.02,r_sp=0.008):
        d_ang = 2*r_ang/nps
        d_sp = 2*r_sp/nps
        ang_iter = np.arange(-r_ang,r_ang+d_ang/2,d_ang)+angle
        sp_iter = np.arange(-r_sp,r_sp+d_sp/2,d_sp)+spacing
        magarr = np.zeros((nps+1,nps+1))
        pharr = np.zeros((nps+1,nps+1))
        for m,ang in enumerate(ang_iter):
            for n,sp in enumerate(sp_iter):
                print (m,n)
                mag, phase = self.getoverlap1(ang,sp,spz)
                if np.isnan(mag):
                    magarr[m,n] = 0.0
                else:
                    magarr[m,n] = mag
                    pharr[m,n] = phase
        figure()
        subplot(211)
        imshow(magarr,interpolation='nearest')
        subplot(212)
        imshow(pharr,interpolation='nearest')
        # get maximum
        k, l = np.where( magarr == magarr.max())
        angmax = k[0]*d_ang - r_ang + angle
        spmax = l[0]*d_sp - r_sp + spacing
        return (angmax,spmax,magarr.max())

    def getoverlapz(self,angle,spacingx,spacingz):
        ''' shift 2nd order data '''
        dx = self.dx / 2
        dz = self.dz / 2
        kx = dx*np.cos(angle)/(spacingx*2)
        ky = dx*np.sin(angle)/(spacingx*2)
        kz = dz/spacingz

        ysh = self.shiftmat(kz,kx,ky).astype(np.complex64)
        otf = self.cufftn((self.psf*ysh))

        ysh = self.shiftmat(0.,kx,ky).astype(np.complex64)
        imgf = self.img_1_0.astype(np.complex64)
        imgf = self.cufftn((imgf*ysh))
        temp = (np.abs(imgf*otf)**2).sum()
        return temp

#    def mapoverlapz(self,angle,spacing,spz,nps=10,r_spz=0.1):
#        d_spz = 2*r_spz/nps
#        spz_iter = np.arange(-r_spz,r_spz+d_spz/2,d_spz)+spz
#        magarr = np.zeros((nps+1))
#        for m,z in enumerate(spz_iter):
#            print (m)
#            temp = self.getoverlapz(angle,spacing,z)
#            if np.isnan(temp):
#                magarr[m] = 0.0
#            else:
#                magarr[m] = temp
#        print(spz_iter)
#        print(magarr)
#        figure()
#        plot(spz_iter,magarr)
#        k = np.where( magarr == magarr.max() )
#        spzmax = k[0]*d_spz - r_spz + spz
#        return (spzmax)
        
    def mapoverlapz(self,angle,spacing,spz,nps=10,r_spz=0.1):
        d_spz = 2*r_spz/nps
        spz_iter = np.arange(-r_spz,r_spz+d_spz/2,d_spz)+spz
        magarr = np.zeros((nps+1))
        for m,z in enumerate(spz_iter):
            print (m)
            mag, phase = self.getoverlap1(angle,spacing,z)
            if np.isnan(mag):
                magarr[m] = 0.0
            else:
                magarr[m] = mag
        print(spz_iter)
        print(magarr)
        figure()
        plot(spz_iter,magarr)
        k = np.where( magarr == magarr.max() )
        spzmax = k[0]*d_spz - r_spz + spz
        return (spzmax)

    def getoverlap2(self,angle,spacingx):
        ''' shift 2nd order data '''
        dx = self.dx / 2
        kx = dx*np.cos(angle)/spacingx
        ky = dx*np.sin(angle)/spacingx
        kz = 0
        
        nxh = self.nx
        nyh = self.ny
        
        ysh = self.shiftmat(kz,kx,ky).astype(np.complex64)
        otf = self.cufftn(self.psf*ysh)
        yshf = np.abs(self.cufftn(ysh))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<nxh):
            sx = sx
        else:
            sx = sx-2*nxh
        if (sy<nyh):
            sy = sy
        else:
            sy = sy-2*nyh 
        zsp = self.zerosuppression( sz, sx, sy)
        otf = otf * zsp
        
        imgf = self.img_2_0
        imgf = self.cufftn(imgf*ysh)
        
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf*imgf0
        wimgf1 = otf0*imgf
        msk = (np.abs(otf0*otf)>cutoff).astype(np.complex64)
        a = np.sum(msk*wimgf1*wimgf0.conj())/np.sum(msk*wimgf0*wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        return mag, phase

    def mapoverlap2(self,angle,spacing,nps=10,r_ang=0.02,r_sp=0.008):
        d_ang = 2*r_ang/nps
        d_sp = 2*r_sp/nps
        ang_iter = np.arange(-r_ang,r_ang+d_ang/2,d_ang)+angle
        sp_iter = np.arange(-r_sp,r_sp+d_sp/2,d_sp)+spacing
        magarr = np.zeros((nps+1,nps+1))
        pharr = np.zeros((nps+1,nps+1))
        for m,ang in enumerate(ang_iter):
            for n,sp in enumerate(sp_iter):
                print (m,n)
                mag, phase = self.getoverlap2(ang,sp)
                if np.isnan(mag):
                    magarr[m,n] = 0.0
                else:
                    magarr[m,n] = mag
                    pharr[m,n] = phase
        figure()
        subplot(211)
        imshow(magarr,interpolation='nearest')
        subplot(212)
        imshow(pharr,interpolation='nearest')
        # get maximum
        k, l = np.where( magarr == magarr.max())
        angmax = k[0]*d_ang - r_ang + angle
        spmax = l[0]*d_sp - r_sp + spacing
        return (angmax,spmax,magarr.max())

    def shift0(self):
        nxh = self.nx
        nyh = self.ny
        nzh = self.nz
        self.otf0 = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        zsp = self.zerosuppression( 0., 0., 0.)
        self.otf0[:] = self.cufftn(self.psf) * zsp
        self.imgf0 = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        self.imgf0 = self.cufftn(self.img_0)
        self.imgf0 = self.imgf0
        tf.imsave(join('otf_0.tif'),self.otf0)
        tf.imsave(join('imgf_0.tif'),self.imgf0)

    def shift1(self,angle,spacingx,spacingz):
        ''' shift 1st order data '''
        dx = self.dx / 2
        dz = self.dz / 2
        nxh = self.nx
        nyh = self.ny
        nzh = self.nz
        kx = dx*np.cos(angle)/(spacingx*2)
        ky = dx*np.sin(angle)/(spacingx*2)
        kz = dz/spacingz
        
        ysh = np.zeros((2,2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        ysh[0,:,:,:] = self.shiftmat(kz,kx,ky).astype(np.complex64)
        ysh[1,:,:,:] = self.shiftmat(kz,-kx,-ky).astype(np.complex64)
        
        otf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        otf[:,:,:] = self.cufftn((self.psf*ysh[0]))
        yshf = np.abs(self.cufftn(ysh[0]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<nxh):
            sx = sx
        else:
            sx = sx-2*nxh
        if (sy<nyh):
            sy = sy
        else:
            sy = sy-2*nyh
        zsp = self.zerosuppression( sz, sx, sy)
        otf = otf * zsp
        tf.imsave(join('otf_1_0.tif'),otf)
        otf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        otf[:,:,:] = self.cufftn((self.psf*ysh[1]))
        yshf = np.abs(self.cufftn(ysh[1]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<nxh):
            sx = sx
        else:
            sx = sx-2*nxh
        if (sy<nyh):
            sy = sy
        else:
            sy = sy-2*nyh
        zsp = self.zerosuppression( sz, sx, sy)
        otf = otf * zsp
        tf.imsave(join('otf_1_1.tif'),otf)
        
        ysh = np.zeros((2,2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        ysh[0,:,:,:] = self.shiftmat(0,kx,ky).astype(np.complex64)
        ysh[1,:,:,:] = self.shiftmat(0,-kx,-ky).astype(np.complex64)
        
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_1_0
        imgf[:,:,:] = self.cufftn(imgf*ysh[0])
        tf.imsave(join('imgf_1_0.tif'),imgf)
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_1_1      
        imgf[:,:,:] = self.cufftn(imgf*ysh[1])
        tf.imsave(join('imgf_1_1.tif'),imgf)

    def shift2(self,angle,spacingx):
        ''' shift 2nd order data '''
        dx = self.dx / 2
        nxh = self.nx
        nyh = self.ny
        nzh = self.nz
        kx = dx*np.cos(angle)/spacingx
        ky = dx*np.sin(angle)/spacingx
        kz = 0
        
        ysh = np.zeros((2,2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        ysh[0,:,:,:] = self.shiftmat(kz,kx,ky).astype(np.complex64)
        ysh[1,:,:,:] = self.shiftmat(kz,-kx,-ky).astype(np.complex64)
        
        otf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        otf[:,:,:] = self.cufftn(self.psf*ysh[0])
        yshf = np.abs(self.cufftn(ysh[0]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<nxh):
            sx = sx
        else:
            sx = sx-2*nxh
        if (sy<nyh):
            sy = sy
        else:
            sy = sy-2*nyh 
        zsp = self.zerosuppression( sz, sx, sy)
        otf = otf * zsp
        tf.imsave(join('otf_2_0.tif'),otf)
        otf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        otf[:,:,:] = self.cufftn(self.psf*ysh[1])
        yshf = np.abs(self.cufftn(ysh[1]))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<nxh):
            sx = sx
        else:
            sx = sx-2*nxh
        if (sy<nyh):
            sy = sy
        else:
            sy = sy-2*nyh 
        zsp = self.zerosuppression( sz, sx, sy)
        otf = otf * zsp
        tf.imsave(join('otf_2_1.tif'),otf)
        
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_2_0
        imgf[:,:,:] = self.cufftn(imgf*ysh[0])
        tf.imsave(join('imgf_2_0.tif'),imgf)
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_2_1        
        imgf[:,:,:] = self.cufftn(imgf*ysh[1])
        tf.imsave(join('imgf_2_1.tif'),imgf)

    def cufftn(self,img):
        data = img.copy()
        d_data = cuda.to_device(data)
        fftplan = Plan.three(CUFFT_C2C, *data.shape)
        fftplan.forward(d_data, d_data)
        d_data.copy_to_host(data)
        return data

    def cuifftn(self,img):
        data = img.copy()
        d_data = cuda.to_device(data)
        fftplan = Plan.three(CUFFT_C2C, *data.shape)
        fftplan.inverse(d_data, d_data)
        d_data.copy_to_host(data)
        return data

    def interp(self,arr):
        nz,nx,ny = arr.shape
        pz = int(nz/2)
        px = int(nx/2)
        py = int(ny/2)
        outarr = np.zeros((2*nz,2*nx,2*ny), dtype=np.complex64)
        arrf = self.cufftn(arr.astype(np.complex64))
        arro = np.pad(np.fft.fftshift(arrf),((pz,pz),(px,px),(py,py)),'constant', constant_values=(0))
        outarr = self.cuifftn(np.fft.fftshift(arro))
        return outarr
    
    def zerosuppression(self,sz,sx,sy):
        x = self.xv
        y = self.yv
        z = self.zv
        g = 1 - self.strength*np.exp(-(np.abs((x-sx)**2)+np.abs((y-sy)**2))+0.*np.abs((z-sz)**2))/(2*(self.fwhm**2))
        g[g<self.thre] = self.minv
        return g

    def window(self,eta):
        nz = self.nz * 2
        nx = self.nx * 2
        ny = self.ny * 2
        wd = np.zeros((nz,nx,ny))
        wind = signal.tukey(nx, alpha=eta, sym=True)
        wx = np.tile(wind,(nx,1))
        wy = wx.swapaxes(0,1)
        w = wx * wy
        for i in range(nz):
            wd[i,:,:,] = w
        return wd
    
    def apod(self):
        rxy = 2.*self.radius_xy
        rz = 2.*self.radius_z
        apo = ( 1 - self.axy * np.sqrt(self.xv**2 + self.yv**2) / rxy )**self.n * ( 1 - self.az * np.sqrt(self.zv**2) / rz )**self.n
        rhxy = np.sqrt(self.xv**2 + self.yv**2 + 0.*self.zv**2)/rxy
        rhz = np.sqrt(0.*self.xv**2 + 0.*self.yv**2 + self.zv**2)/rz
        msk_xy = (rhxy<=1.0).astype(np.float64)
        msk_z = (rhz<=1.0).astype(np.float64)
        msk = msk_xy * msk_z
        apodiz = apo * msk
        return apodiz

    def recon1(self,phase1,mag1,phase2,mag2):
        # construct 1 angle
        nx = 2*self.nx
        ny = 2*self.ny
        nz = 2*self.nz
        mu = self.mu
        ph1 = mag1*np.exp(1j*phase1)
        ph2 = mag2*np.exp(1j*phase2)
        
        imgf = np.zeros((nz,nx,nx),dtype=np.complex64)
        otf = np.zeros((nz,nx,nx),dtype=np.complex64)        
        
        Snum = np.zeros((nz,nx,ny),dtype=np.complex64)
        Sden = np.zeros((nz,nx,ny),dtype=np.complex64)
        Sden += mu**2
        # 0th order
        imgf = tf.imread(join('imgf_0.tif'))
        otf = tf.imread(join('otf_0.tif'))
        Snum += otf.conj()*imgf
        Sden += np.abs(otf)**2
        # +1st order
        imgf = tf.imread(join('imgf_1_0.tif'))
        otf = tf.imread(join('otf_1_0.tif'))
        Snum += ph1*otf.conj()*imgf
        Sden += np.abs(otf)**2
        # -1 order
        imgf = tf.imread(join('imgf_1_1.tif'))
        otf = tf.imread(join('otf_1_1.tif'))
        Snum += ph1.conj()*otf.conj()*imgf
        Sden += np.abs(otf)**2
        # +2nd order
        imgf = tf.imread(join('imgf_2_0.tif'))
        otf = tf.imread(join('otf_2_0.tif'))
        Snum += ph2*otf.conj()*imgf
        Sden += np.abs(otf)**2
        # -2nd order
        imgf = tf.imread(join('imgf_2_1.tif'))
        otf = tf.imread(join('otf_2_1.tif'))
        Snum += ph2.conj()*otf.conj()*imgf
        Sden += np.abs(otf)**2
        # finish
        S = Snum/Sden
        self.Snum = Snum
        self.Sden = Sden
        self.finalimage = fftshift(self.cuifftn(S))
        return True

    def recon_add(self,phase1,mag1,phase2,mag2):
        # construct 1 angle
        nx = 2*self.nx
        ny = 2*self.ny
        nz = 2*self.nz
        ph1 = mag1*np.exp(1j*phase1)
        ph2 = mag2*np.exp(1j*phase2)

        imgf = np.zeros((nz,nx,ny),dtype=np.complex64)
        otf = np.zeros((nz,nx,ny),dtype=np.complex64)        

        # 0th order
        imgf = tf.imread(join('imgf_0.tif'))
        otf = tf.imread(join('otf_0.tif'))
        self.Snum += otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # +1st order
        imgf = tf.imread(join('imgf_1_0.tif'))
        otf = tf.imread(join('otf_1_0.tif'))
        self.Snum += ph1*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # -1 order
        imgf = tf.imread(join('imgf_1_1.tif'))
        otf = tf.imread(join('otf_1_1.tif'))
        self.Snum += ph1.conj()*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # +2nd order
        imgf = tf.imread(join('imgf_2_0.tif'))
        otf = tf.imread(join('otf_2_0.tif'))
        self.Snum += ph2*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # -2nd order
        imgf = tf.imread(join('imgf_2_1.tif'))
        otf = tf.imread(join('otf_2_1.tif'))
        self.Snum += ph2.conj()*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # finish
        S = self.Snum/self.Sden 
        self.finalimage = fftshift(self.cuifftn(S))
        return True
        