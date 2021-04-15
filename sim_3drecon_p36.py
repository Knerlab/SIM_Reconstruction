# -*- coding: utf-8 -*-
"""
3D structured illumination microscopy image reconstruction algorithm
@copyright, Ruizhe Lin and Peter Kner, University of Georgia, 2019

"""

import os
temppath = r'C:/Users/rl74173/Documents/PythonScripts/temp'
join = lambda fn: os.path.join(temppath,fn)

import tifffile as tf
import psfsim36
from pylab import imshow, subplot, figure, plot
import numpy as np
from cupyfft import fftn, ifftn
from scipy.fftpack import fftshift
from scipy import signal

class si3D(object):

    def __init__(self, fnd, nph, nangles, wavelength, na):
        self.img_stack = tf.imread(fnd) # order of images should be phases, angles, zslices
        # self.img_stack = self.subback(self.img_stack)
        self.img_stack = np.pad(self.img_stack, ((2*nph*nangles,2*nph*nangles),(0,0),(0,0)),'constant', constant_values=(0))
        print('Image stack loaded succefully')
        nz,nx,ny = self.img_stack.shape
        self.nz = int(nz/nph/nangles)
        self.nx = nx
        self.ny = ny
        self.mu = 1e-2 # Wiener parameter
        self.wl = wavelength # in microns
        self.cutoff = 1e-3 # remove noise below this relative value in freq. space
        self.na = na # numerical aperture
        self.dx = 0.089 # pixel size in microns
        self.dz = 0.2 # z step in microns
        self.nphases = nph
        self.norders = 5
        self.dpx = 1/((self.nx*2.)*(self.dx/2.)) # calculate pixel size in frequency space
        self.dpz = 1/((self.nz*2.)*(self.dz/2.)) # calculate axial pixel size in frequency space
        self.radius_xy = (2*self.na/self.wl)/self.dpx # NA in pixels
        self.radius_z = ((self.na**2)/(2*self.wl))/self.dpz
        self.strength = 1.
        self.sigma = 8.
        self.eta = 0.08
        self.expn = 1.
        self.axy = 0.8
        self.az = 0.8
        self.zoa = 10e-2
        self.nangle = 0
        self.psf = self.getpsf()
        self.sepmat = self.sepmatrix()
        self.meshgrid()
        self.winf = self.window(self.eta)
        self.apd = self.apod()
        self.img_stack = self.img_stack.reshape(self.nz,nangles,nph,nx,ny).swapaxes(0,1).swapaxes(1,2)

    def subback(self,img):
        hist, bin_edges  = np.histogram(img, bins=np.arange(img.min(),img.max(),256))
        ind = np.where(hist == hist.max())
        bg = bin_edges[np.max(ind[0]*2)]
        img[img<=bg] = 0.
        img[img>bg] = img[img>bg] - bg
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
        ''' multiply by exponential in real space to create shift in frequency space  '''
        zv = self.zv
        xv = self.xv
        yv = self.yv
        a = np.exp(2j*np.pi*(kx*xv+ky*yv))*np.cos(2*np.pi*kz*zv)
        return a

    def sepmatrix(self):
        ''' create matrix to separate orders from raw data '''
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
        ''' get simulated PSF for OTF calculations '''
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
        
    def separate(self, nang=0):
        ''' separate raw sim data into orders '''
        self.nangle = nang
        npx = self.nz*self.nx*self.ny
        nz = self.nz
        nx = self.nx
        ny = self.ny
        out = np.dot(self.sepmat,self.img_stack[nang].reshape(self.nphases,npx))
        self.img_0 = fftshift(self.interp(out[0].reshape(nz,nx,ny))*self.winf)
        self.img_1_0 = fftshift(self.interp((out[1]+1j*out[2]).reshape(nz,nx,ny))*self.winf)
        self.img_1_1 = fftshift(self.interp((out[1]-1j*out[2]).reshape(nz,nx,ny))*self.winf)
        self.img_2_0 = fftshift(self.interp((out[3]+1j*out[4]).reshape(nz,nx,ny))*self.winf)
        self.img_2_1 = fftshift(self.interp((out[3]-1j*out[4]).reshape(nz,nx,ny))*self.winf)

    def getoverlap1(self,angle,spacingx,spacingz,plot=False):
        ''' calculate overlap between 0th and 1st order in xy '''
        dx = self.dx / 2
        dz = self.dz / 2
        kx = dx*np.cos(angle)/(spacingx*2)
        ky = dx*np.sin(angle)/(spacingx*2)
        kz = dz/spacingz     
      
        ysh = np.exp(2j*np.pi*(kx*self.xv+ky*self.yv+kz*self.zv))
        otf = fftn((self.psf*ysh))
        
        nxh = self.nx
        nyh = self.ny
        yshf = np.abs(fftn(ysh))
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
        
        ysh = np.exp(2j*np.pi*(kx*self.xv+ky*self.yv+0.*self.zv))
        imgf = self.img_1_0.astype(np.complex64)
        imgf = fftn((imgf*ysh))
        
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf*imgf0
        wimgf1 = otf0*imgf
        msk = (np.abs(otf0*otf)>cutoff).astype(np.complex64)
        if plot==True:
            tf.imshow(np.abs((msk*wimgf1*wimgf0.conj())/(msk*wimgf0*wimgf0.conj())))
            tf.imshow(np.angle((msk*wimgf1*wimgf0.conj())/(msk*wimgf0*wimgf0.conj())))
        a = np.sum(msk*wimgf1*wimgf0.conj())/np.sum(msk*wimgf0*wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        return mag, phase

    def mapoverlap1(self,angle,spacing,spz,nps=10,r_ang=0.02,r_sp=0.008):
        ''' find optimal spacing and angle for first order in xy '''
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
        ''' find optimal spacing in z direction for 1st order '''
        dx = self.dx / 2
        dz = self.dz / 2
        kx = dx*np.cos(angle)/(spacingx*2)
        ky = dx*np.sin(angle)/(spacingx*2)
        kz = dz/spacingz

        ysh = np.exp(2j*np.pi*(kx*self.xv+ky*self.yv+kz*self.zv))
        otf = fftn((self.psf*ysh))
        yshf = np.abs(fftn(ysh))
        nxh = self.nx
        nyh = self.ny
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
        imgf = fftn((imgf*ysh))
        temp = (np.abs(imgf*otf)**2).sum()
        return temp

    def mapoverlapz(self,angle,spacing,spz,nps=10,r_spz=0.1):
        ''' Find optimal spacing in axial direction for 1st order '''
        d_spz = 2*r_spz/nps
        spz_iter = np.arange(-r_spz,r_spz+d_spz/2,d_spz)+spz
        magarr = np.zeros((nps+1))
        for m,z in enumerate(spz_iter):
            print (m)
            temp = self.getoverlapz(angle,spacing,z)
            if np.isnan(temp):
                magarr[m] = 0.0
            else:
                magarr[m] = temp
        print(spz_iter)
        print(magarr)
        figure()
        plot(spz_iter,magarr)
        k = np.where( magarr == magarr.max() )
        spzmax = k[0]*d_spz - r_spz + spz
        return (spzmax)

    def getoverlap2(self,angle,spacingx,plot=False):
        ''' calculate overlop between 2nd and 0th order '''
        dx = self.dx / 2
        kx = dx*np.cos(angle)/spacingx
        ky = dx*np.sin(angle)/spacingx
        kz = 0
        
        ysh = np.exp(2j*np.pi*(kx*self.xv+ky*self.yv+kz*self.zv))
        otf = fftn(self.psf*ysh)
        
        nxh = self.nx
        nyh = self.ny
        yshf = np.abs(fftn(ysh))
        sz, sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<nxh):
            sx = sx
        else:
            sx = sx-2*nxh
        if (sy<nyh):
            sy = sy
        else:
            sy = sy-2*nyh 
        zsp = self.zerosuppression(sz, sx, sy)
        otf = otf * zsp
        
        imgf = self.img_2_0
        imgf = fftn(imgf*ysh)
        
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf*imgf0
        wimgf1 = otf0*imgf
        msk = (np.abs(otf0*otf)>cutoff).astype(np.complex64)
        if plot==True:
            tf.imshow(np.abs((msk*wimgf1*wimgf0.conj())/(msk*wimgf0*wimgf0.conj())))
            tf.imshow(np.angle((msk*wimgf1*wimgf0.conj())/(msk*wimgf0*wimgf0.conj())))
        a = np.sum(msk*wimgf1*wimgf0.conj())/np.sum(msk*wimgf0*wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        return mag, phase

    def mapoverlap2(self,angle,spacing,nps=10,r_ang=0.02,r_sp=0.008):
        ''' find optimal spacing and angle for 2nd order '''
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

    def shift0(self,plot=False):
        ''' pad 0th order data, create 0th order OTF '''
        nxh = self.nx
        nyh = self.ny
        nzh = self.nz
        self.otf0 = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        zsp = self.zerosuppression( 0., 0., 0.)
        self.otf0[:] = fftn(self.psf) * zsp
        self.imgf0 = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        self.imgf0 = fftn(self.img_0)
        self.imgf0 = self.imgf0
        tf.imsave(join('otf_0.tif'),self.otf0)
        tf.imsave(join('imgf_0.tif'),self.imgf0)
        if plot==True:
            tf.imshow(np.abs(fftshift(self.otf0)), photometric='minisblack',title='Angle %d _ 0 order OTF'%self.nangle)
            tf.imshow(np.abs(fftshift(self.imgf0)), photometric='minisblack',title='Angle %d _ 0 order frequency spectrum'%self.nangle)

    def shift1(self,angle,spacingx,spacingz,plot=False):
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
        otf[:,:,:] = fftn((self.psf*ysh[0]))
        yshf = np.abs(fftn(ysh[0]))
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
        if plot==True:
            tf.imshow(np.abs(fftshift(otf)),photometric='minisblack',title='Angle %d _ 1st order +1 OTF'%self.nangle)
        otf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        otf[:,:,:] = fftn((self.psf*ysh[1]))
        yshf = np.abs(fftn(ysh[1]))
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
        if plot==True:
            tf.imshow(np.abs(fftshift(otf)),photometric='minisblack',title='Angle %d _ 1st order -1 OTF'%self.nangle)

        ysh = np.zeros((2,2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        ysh[0,:,:,:] = self.shiftmat(0,kx,ky).astype(np.complex64)
        ysh[1,:,:,:] = self.shiftmat(0,-kx,-ky).astype(np.complex64)
        
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_1_0
        imgf[:,:,:] = fftn(imgf*ysh[0])
        tf.imsave(join('imgf_1_0.tif'),imgf)
        if plot==True:
            tf.imshow(np.abs(fftshift(imgf)),photometric='minisblack',title='Angle %d _ 1st order +1 frequency spectrum'%self.nangle)
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_1_1      
        imgf[:,:,:] = fftn(imgf*ysh[1])
        tf.imsave(join('imgf_1_1.tif'),imgf)       
        if plot==True:
            tf.imshow(np.abs(fftshift(imgf)),photometric='minisblack',title='Angle %d _ 1st order -1 frequency spectrum'%self.nangle)

    def shift2(self,angle,spacingx,plot=False):
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
        otf[:,:,:] = fftn(self.psf*ysh[0])
        yshf = np.abs(fftn(ysh[0]))
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
        if plot==True:
            tf.imshow(np.abs(fftshift(otf)),photometric='minisblack',title='Angle %d _ 2nd order +1 OTF'%self.nangle)
        otf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        otf[:,:,:] = fftn(self.psf*ysh[1])
        yshf = np.abs(fftn(ysh[1]))
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
        if plot==True:
            tf.imshow(np.abs(fftshift(otf)),photometric='minisblack',title='Angle %d _ 2nd order -1 OTF'%self.nangle)
            
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_2_0
        imgf[:,:,:] = fftn(imgf*ysh[0])
        tf.imsave(join('imgf_2_0.tif'),imgf)
        if plot==True:
            tf.imshow(np.abs(fftshift(imgf)),photometric='minisblack',title='Angle %d _ 2nd order +1 frequency spectrum'%self.nangle)
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf[:,:,:] = self.img_2_1        
        imgf[:,:,:] = fftn(imgf*ysh[1])
        tf.imsave(join('imgf_2_1.tif'),imgf)
        if plot==True:
            tf.imshow(np.abs(fftshift(imgf)),photometric='minisblack',title='Angle %d _ 2nd order -1 frequency spectrum'%self.nangle)

    def interp(self,arr):
        ''' interpolate by padding in frequency space '''
        nz,nx,ny = arr.shape
        outarr = np.zeros((2*nz,2*nx,2*ny), dtype=arr.dtype)
        arrf = fftn(arr)
        arro = self.pad(arrf)
        outarr = ifftn(arro)
        return outarr

    def pad(self,arr):
        ''' pad with zeros keeping zero frequency at corner '''
        nz,nx,ny = arr.shape
        out = np.zeros((2*nz,2*nx,2*nx),arr.dtype)
        nxh = np.int(nx/2)
        if nz%2==0:
            nzh = np.int(nz/2)
            out[:nzh,:nxh,:nxh] = arr[:nzh,:nxh,:nxh]
            out[:nzh,:nxh,3*nxh:4*nxh] = arr[:nzh,:nxh,nxh:nx]
            out[:nzh,3*nxh:4*nxh,:nxh] = arr[:nzh,nxh:nx,:nxh]
            out[:nzh,3*nxh:4*nxh,3*nxh:4*nxh] = arr[:nzh,nxh:nx,nxh:nx]
            out[3*nzh:4*nzh,:nxh,:nxh] = arr[nzh:nz,:nxh,:nxh]
            out[3*nzh:4*nzh,:nxh,3*nxh:4*nxh] = arr[nzh:nz,:nxh,nxh:nx]
            out[3*nzh:4*nzh,3*nxh:4*nxh,:nxh] = arr[nzh:nz,nxh:nx,:nxh]
            out[3*nzh:4*nzh,3*nxh:4*nxh,3*nxh:4*nxh] = arr[nzh:nz,nxh:nx,nxh:nx]
        else:
            nzh = np.int(nz/2)
            out[:nzh,:nxh,:nxh] = arr[:nzh,:nxh,:nxh]
            out[:nzh,:nxh,3*nxh:4*nxh] = arr[:nzh,:nxh,nxh:nx]
            out[:nzh,3*nxh:4*nxh,:nxh] = arr[:nzh,nxh:nx,:nxh]
            out[:nzh,3*nxh:4*nxh,3*nxh:4*nxh] = arr[:nzh,nxh:nx,nxh:nx]
            out[(3*nzh+1):(2*nz),:nxh,:nxh] = arr[nzh:nz,:nxh,:nxh]
            out[(3*nzh+1):(2*nz),:nxh,3*nxh:4*nxh] = arr[nzh:nz,:nxh,nxh:nx]
            out[(3*nzh+1):(2*nz),3*nxh:4*nxh,:nxh] = arr[nzh:nz,nxh:nx,:nxh]
            out[(3*nzh+1):(2*nz),3*nxh:4*nxh,3*nxh:4*nxh] = arr[nzh:nz,nxh:nx,nxh:nx]
        return out
    
    def zerosuppression(self,sz,sx,sy):
        ''' suppress zero frequency in SIM reconstruction '''
        x = self.xv
        y = self.yv
        z = self.zv
        g = 1 - self.strength * np.exp(-((x-sx)**2.+(y-sy)**2.+0.*(z-sz)**2.)/(2.*self.sigma**2.))
        g[g<0.5] = 0.0
        g[g>=0.5] = 1.0
        return g

    def window(self,eta):
        ''' windowing to avoid edge effects in FFT '''
        nz = self.nz * 2
        nx = self.nx * 2
        ny = self.ny * 2
        wd = np.zeros((nz,nx,ny))
        wind = signal.tukey(nx, alpha=eta, sym=True)
        wz = signal.tukey(nz, alpha=eta, sym=True)
        wx = np.tile(wind,(nx,1))
        wy = wx.swapaxes(0,1)
        w = wx * wy
        for i in range(nz):
            wd[i,:,:,] = w * wz[i]
        return wd
    
    def apod(self):
        ''' apodization to avoid ringing '''
        rxy = 2.*self.radius_xy
        rz = 2.*self.radius_z
        apo = ( 1 - self.axy * np.sqrt(self.xv**2 + self.yv**2) / rxy )**self.expn * ( 1 - self.az * np.sqrt(self.zv**2) / rz )**self.expn
        rhxy = np.sqrt(self.xv**2 + self.yv**2 + 0.*self.zv**2)/rxy
        rhz = np.sqrt(0.*self.xv**2 + 0.*self.yv**2 + self.zv**2)/rz
        msk_xy = (rhxy<=1.0).astype(np.float64)
        msk_z = (rhz<=1.0).astype(np.float64)
        msk = msk_xy * msk_z
        apodiz = apo * msk
        return apodiz

    def recon1(self,phase1,mag1,phase2,mag2):
        ''' SIM reconstruction of one angle '''
        # construct 1 angle
        nx = 2*self.nx
        ny = 2*self.ny
        nz = 2*self.nz
        mu = self.mu
        ph0 = self.zoa
        ph1 = mag1*np.exp(1j*phase1)
        ph2 = mag2*np.exp(1j*phase2)
        
        imgf = np.zeros((nz,nx,nx),dtype=np.complex64)
        otf = np.zeros((nz,nx,nx),dtype=np.complex64)        
        
        self.Snum = np.zeros((nz,nx,ny),dtype=np.complex64)
        self.Sden = np.zeros((nz,nx,ny),dtype=np.complex64)
        self.Sden += mu**2
        # 0th order
        imgf = tf.imread(join('imgf_0.tif'))
        tf.imsave('angle%d_imgf_0.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_0.tif'))
        tf.imsave('angle%d_otf_0.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph0 * otf.conj() * imgf
        self.Sden += np.abs(otf)**2
        # +1st order
        imgf = tf.imread(join('imgf_1_0.tif'))
        tf.imsave('angle%d_imgf_1_0.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_1_0.tif'))
        tf.imsave('angle%d_otf_1_0.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph1*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # -1 order
        imgf = tf.imread(join('imgf_1_1.tif'))
        tf.imsave('angle%d_imgf_1_1.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_1_1.tif'))
        tf.imsave('angle%d_otf_1_1.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph1.conj()*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # +2nd order
        imgf = tf.imread(join('imgf_2_0.tif'))
        tf.imsave('angle%d_imgf_2_0.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_2_0.tif'))
        tf.imsave('angle%d_otf_2_0.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph2*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # -2nd order
        imgf = tf.imread(join('imgf_2_1.tif'))
        tf.imsave('angle%d_imgf_2_1.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_2_1.tif'))
        tf.imsave('angle%d_otf_2_1.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph2.conj()*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # # finish
        # S = self.Snum/self.Sden
        # self.finalimage = ifftn(S)
        return True

    def recon_add(self,phase1,mag1,phase2,mag2):
        ''' add additional angles to SIM reconstruction '''
        # construct 1 angle
        nx = 2*self.nx
        ny = 2*self.ny
        nz = 2*self.nz
        ph1 = mag1*np.exp(1j*phase1)
        ph2 = mag2*np.exp(1j*phase2)
        ph0 = self.zoa
        
        imgf = np.zeros((nz,nx,ny),dtype=np.complex64)
        otf = np.zeros((nz,nx,ny),dtype=np.complex64)        
        # 0th order
        imgf = tf.imread(join('imgf_0.tif'))
        tf.imsave('angle%d_imgf_0.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_0.tif'))
        tf.imsave('angle%d_otf_0.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph0 * otf.conj() * imgf
        self.Sden += np.abs(otf)**2
        # +1st order
        imgf = tf.imread(join('imgf_1_0.tif'))
        tf.imsave('angle%d_imgf_1_0.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_1_0.tif'))
        tf.imsave('angle%d_otf_1_0.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph1*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # -1 order
        imgf = tf.imread(join('imgf_1_1.tif'))
        tf.imsave('angle%d_imgf_1_1.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_1_1.tif'))
        tf.imsave('angle%d_otf_1_1.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph1.conj()*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # +2nd order
        imgf = tf.imread(join('imgf_2_0.tif'))
        tf.imsave('angle%d_imgf_2_0.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_2_0.tif'))
        tf.imsave('angle%d_otf_2_0.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph2*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # -2nd order
        imgf = tf.imread(join('imgf_2_1.tif'))
        tf.imsave('angle%d_imgf_2_1.tif'%self.nangle,np.abs(np.fft.fftshift(imgf)).astype(np.float32),photometric='minisblack')
        otf = tf.imread(join('otf_2_1.tif'))
        tf.imsave('angle%d_otf_2_1.tif'%self.nangle,np.abs(np.fft.fftshift(otf)).astype(np.float32),photometric='minisblack')
        self.Snum += ph2.conj()*otf.conj()*imgf
        self.Sden += np.abs(otf)**2
        # # finish
        # S = self.Snum/self.Sden 
        # self.finalimage = fftshift(ifftn(S))
        return True
