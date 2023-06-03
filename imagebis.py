############################################
#CE CODE N'EST PAS DE MOI ET SERT JUSTE A FLOUTER L'IMAGE
############################################

import numpy as np
from scipy.fftpack import dct, idct

def dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct2(coefficient):
    return idct(idct(coefficient.T, norm='ortho').T, norm='ortho')

def create_kernel(kernel_size, kernel_std):
    x = np.concatenate((np.arange(0, kernel_size / 2), np.arange(-kernel_size / 2, 0)))
    [Y, X] = np.meshgrid(x, x)
    kernel = np.exp((-X**2 - Y**2) / (2 * kernel_std**2))
    kernel = kernel / sum(kernel.flatten())
    return kernel

def gaussian_kernel(kernel_size, kernel_std, im_shape):
    kernel = create_kernel(kernel_size, kernel_std)
    padded_kernel = np.zeros(im_shape)
    padded_kernel[0:kernel.shape[0], 0:kernel.shape[1]] = kernel
    e = np.zeros(im_shape)
    e[0,0] = 1
    i = int(np.floor(kernel.shape[0]/2)+1)
    j = int(np.floor(kernel.shape[1]/2)+1)
    center = np.array([i, j])
    m = im_shape[0]
    n = im_shape[1]
    k = min(i-1,m-i,j-1,n-j)

    PP = padded_kernel[i-k-1:i+k, j-k-1:j+k] 
    Z1 = np.diag(np.ones((k+1)),k)
    Z2 = np.diag(np.ones((k)),k+1)
    PP = Z1@PP@Z1.T + Z1@PP@Z2.T + Z2@PP@Z1.T + Z2@PP@Z2.T

    Ps = np.zeros(im_shape)
    Ps[0:2*k+1, 0:2*k+1] = PP
    kernel_fourier = dct2(Ps)/dct2(e)
    return kernel_fourier

kernel_size = 3
kernel_std = 1
im_shape = (128,128)
kernel_fourier = gaussian_kernel(kernel_size, kernel_std, im_shape)

def flou(x): # La fonction qui permet d'évaluer A@x
    return np.real(idct2(dct2(x) * kernel_fourier))

Lipschitz_flou = np.max(np.abs(kernel_fourier)) # Ceci est la constante de Lipschitz du gradient de la fonction quadratique
