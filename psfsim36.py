# -*- coding: utf-8 -*-
"""

This class generates simulated PSFs and OTFs
@copyright, Ruizhe Lin and Peter Kner, University of Georgia, 2019

"""

import numpy as np
import Utility36 as U
import Zernike36 as Z

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
pi = np.pi
from scipy.special import j1

class psf(object):
    ''' code for simulating the point spread function
        main function is get3Dpsf() '''

    def __init__(self):
        self.wl = 0.515 # wavelength in microns
        self.na = 1.2 # numerical aperture
        self.n2 = 1.512 # index at point
        self.dx = 0.089 # pixel size in microns
        self.nx = 256 # size of region
        self.dp = 1/(self.nx*self.dx) # pixel size in frequency space (pupil)
        self.radius = (self.na/self.wl)/self.dp # radius of pupil (NA/lambda) in pixels
        self.zarr = np.zeros(15)
        
    def __del__(self):
        pass
    
    def setParams(self,wl=None,na=None,dx=None,nx=None):
        if wl != None:
            self.wl = wl
        if na != None:
            self.na = na
        if dx != None:
            self.dx = dx
        if nx != None:
            self.nx = nx
        self.dp = 1/(self.nx*self.dx)
        self.radius = (self.na/self.wl)/self.dp

    def getFlatWF(self):
        self.bpp = U.discArray((self.nx,self.nx),self.radius)
    
    def getZArrWF(self,zarr):
        self.zarr = zarr
        msk = U.discArray((self.nx,self.nx),self.radius)
        ph = np.zeros((self.nx,self.nx),dtype=np.float32)
        for j,m in enumerate(zarr):
            ph += m*Z.Zm(j,rad=self.radius,orig=None,Nx=self.nx)
        self.bpp = msk*np.exp(1j*ph)
        
    def focusmode(self,d):
        x = np.arange(-self.nx/2,self.nx/2,1)
        X,Y = np.meshgrid(x,x)
        rho = np.sqrt(X**2 + Y**2)/self.radius
        msk = (rho<=1.0).astype(np.float64)
        wf = msk*(self.n2*d/self.wl)*np.sqrt(1-(self.na*msk*rho/self.n2)**2)
        return wf
        
    def get3Dpsf(self,start,stop,step):
        nsteps = int( (stop-start)/step+1 )
        zarr = np.linspace(start/step,stop/step,nsteps).astype(np.int64)
        zarr = zarr[0:nsteps-1]
        zarr = np.roll(zarr,int((nsteps-1)/2))
        self.stack = np.zeros((nsteps-1,self.nx,self.nx))
        for m,z in enumerate(zarr):
            ph = self.focusmode(z*step)
            wf = self.bpp*np.exp(2j*pi*ph)
            self.stack[m] = np.abs(fft2(wf))**2
        return True
        
    def getOTF3D(self):
        self.otf3D = np.fft.fftn(self.stack)
        self.otf3D = self.otf3D/self.otf3D[0,0,0]
        return True
        
    def otf2d(self):
        nx = self.nx
        ds = (self.wl/self.na)/self.dx/self.nx
        g = lambda r: np.select([(ds*r<2)], [(2*np.arccos(ds*r/2)-np.sin(2*np.arccos(ds*r/2)))/np.pi], 0.0)
        otf =  U.radialArray((nx,nx), g, origin=None)
        self.otf = U.radialArray((nx,nx), g, origin=(0,0))
        return otf
        
    def otf2dstok(self,z):
        ''' w is defocus in microns
            approximation from Stokseth paper '''
        sina = self.na/self.n2
        w = z*(1-np.sqrt(1-sina**2))
        if w==0:
            otf = self.otf2d()
        else:
            nx = self.nx
            ds = (self.wl/self.na)/self.dx/self.nx
            da = 4*pi*w*ds/self.wl
            g = lambda r: np.select([(r<=0.8),(ds*r<2)],
                [1.0,2*(1-0.69*ds*r+0.0076*(ds*r)**2+0.043*(ds*r)**3)*
                (j1(da*r-0.5*(da*r)*(ds*r))/(da*r-0.5*(da*r)*(ds*r)))],
                0.0)
            otf =  U.radialArray((nx,nx), g, origin=None)
            self.otf = U.radialArray((nx,nx), g, origin=(0,0))
        return otf

    def otf2daberr(self,zarr,w):
        #self.getZArrWF(zarr)
        ph = self.focusmode(w)
        wf = self.bpp*np.exp(2j*pi*ph)
        t = np.abs(fftshift(fft2(wf)))**2
        #Y.view(t)
        otf = ifft2(t)
        otf = otf/otf[0,0]
        self.otf = otf
        return otf
        
    def get3Dpsf2Obj(self,start,stop,step):
        nsteps = (stop-start)/step + 1
        zarr = np.linspace(start,stop,nsteps)
        self.stack = np.zeros((nsteps,self.nx,self.nx))
        for m,z in enumerate(zarr):
            ph1 = self.focusmode(z)
            wf1 = self.bpp*np.exp(2j*pi*ph1)
            ph2 = self.focusmode(-z)
            wf2 = self.bpp*np.exp(2j*pi*ph2)
            self.stack[m] = np.abs(fftshift(fft2(wf1))+fftshift(fft2(wf2)))**2
        return True