# -*- coding: utf-8 -*-
"""

These functions are utilities used in different packages
@copyright, Ruizhe Lin and Peter Kner, University of Georgia, 2019

"""

import numpy as np
import Zernike36 as Z

def discArray(shape=(128,128),radius=64,origin=None,dtype=np.float64):
    nx = shape[0]
    ny = shape[1]
    ox = nx/2
    oy = ny/2
    x = np.linspace(-ox,ox-1,nx)
    y = np.linspace(-oy,oy-1,ny)
    X,Y = np.meshgrid(x,y)
    rho = np.sqrt(X**2 + Y**2)
    disc = (rho<radius).astype(dtype)
    if not origin==None:
        s0 = origin[0]-int(nx/2)
        s1 = origin[1]-int(ny/2)
        disc = np.roll(np.roll(disc,int(s0),0),int(s1),1)
    return disc

def radialArray(shape=(128,128), func=None, origin=None, dtype=np.float64):
    nx = shape[0]
    ny = shape[1]
    ox = nx/2
    oy = ny/2
    x = np.linspace(-ox,nx-ox,nx)
    y = np.linspace(-oy,ny-oy,ny)
    X,Y = np.meshgrid(x,y)
    rho = np.sqrt(X**2 + Y**2)
    rarr = func(rho)
    if not origin==None:
        s0 = origin[0]-nx/2
        s1 = origin[1]-ny/2
        rarr = np.roll(np.roll(rarr,int(s0),0),int(s1),1)
    return rarr
    
def shift(arr,shifts=None):
    if shifts == None:
        shifts = ( np.array(arr.shape)/2 ).astype(np.uint16)
    if len(arr.shape)==len(shifts):
        for m,p in enumerate(shifts):
            arr = np.roll(arr,int(p),m)
    return arr
    
def buildphiZ(phi,shape,radius):
    nx = shape[0]
    pupil = np.zeros(shape)
    for m,amp in enumerate(phi):
        pupil = pupil + amp*Z.Zm(m,radius,None,nx)
    return pupil
