'''
This file is based on the codes for the P2S module and the simulator. 

Z. Mao, N. Chimitt, and S. H. Chan, "Accerlerating Atmospheric Turbulence 
Simulation via Learned Phase-to-Space Transform", ICCV 2021

Arxiv: https://arxiv.org/abs/2107.11627

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
Copyright 2021
Purdue University, West Lafayette, IN, USA
'''

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

class Simulator(nn.Module):
    '''
    Class variables:  
        turb_params -  All paramters to generate the random field of zernike coefficients
        data_path   -  Path where the model, PSF dictionary, and integration of bessel functions are stored
    '''
    def __init__(self, turb_params, rf_range=512, data_path='./utils'):
        super().__init__()
        self.gh, self.gw = rf_range, rf_range
        self.h, self.w = turb_params['img_size']
        if self.h > self.gh:
            self.gh = self.h
        if self.w > self.gw:
            self.gw = self.w
        # self.device = torch.device('cuda:0,1') if torch.cuda.is_available() else torch.device('CPU')
        # load the phase to space mapping model
        self.mapping = _P2S()
        self.mapping.load_state_dict(torch.load(os.path.join(data_path, 'P2S_model.pt')))
        # load the basis of point spread functions
        self.dict_psf = torch.load(os.path.join(data_path, 'dictionary.pt'))
        self.mu = self.dict_psf['mu'].view(1, 1, 33, 33).type(torch.float32).cuda()
        self.basis_psf = self.dict_psf['dictionary'].unsqueeze(1).type(torch.float32).cuda()
        self.offset = torch.tensor([31, 31]).cuda() #?
        # define key parameters for simulation
        self.turb_params = turb_params
        self.D = turb_params['D']
        self.r0 = turb_params['r0']
        self.Dr0 = torch.tensor(self.D/self.r0).type(torch.float32).cuda()
        self.R_z, self.sqrt_psd = self._corr_mat(turb_params['corr'])
        self.I0_I2 = torch.load(os.path.join(data_path, 'I0_I2.pt'))
        self.const, self.S_half = self._tilt_mat()
        # define coordination
        yy, xx = torch.meshgrid(torch.arange(0, self.h),torch.arange(0, self.w))
        self.pixel_pos = torch.stack((xx, yy), -1).unsqueeze(0).type(torch.float32).cuda()

    def change_param(self, new_turb_params):
        # call this to generate new correlation matrices for turbulence patterns
        if self.turb_params['corr'] == new_turb_params['corr']:
            self.turb_params = new_turb_params
            self.D = new_turb_params['D']
            self.r0 = new_turb_params['r0']
            self.Dr0 = torch.tensor(self.D/self.r0).type(torch.float32).cuda()
            self.const, self.S_half = self._tilt_mat()
        else:
            self.turb_params = new_turb_params
            self.D = new_turb_params['D']
            self.r0 = new_turb_params['r0']
            self.Dr0 = torch.tensor(self.D/self.r0).type(torch.float32).cuda()
            self.const, self.S_half = self._tilt_mat()
            self.R_z, self.sqrt_psd = self._corr_mat(new_turb_params['corr'])
        
    def _corr_mat(self, corr, num_zern=36):
        '''
        This function generates the correlation matrix for zernike coefficients associated with
            higher-order abberations for the entire image
        Input: 
            corr       -  Correlation strength. suggested range: (-1 ~ -0.01), with -1 has the
                        weakest correlation and -0.01 has the strongest.
            num_zern   -  number of zernike coefficients. Default: 36
        '''
        # subC = _nollCovMat(num_zern, self.D, self.r0).cuda()
        subC = _nollCovMat(num_zern, 1, 1).cuda()  # current setting
        e_val, e_vec = torch.linalg.eig(subC)
        R_z = torch.real(e_vec * torch.sqrt(e_val))
        h = torch.linspace(0, self.gh-1, self.gh).cuda() - self.gh/2
        w = torch.linspace(0, self.gw-1, self.gw).cuda() - self.gw/2
        yv, xv = torch.meshgrid(h, w)
        # dist = torch.exp(corr*(xv.abs()+yv.abs()))
        dist = torch.exp(corr*(xv**2 + yv**2)**0.5)
        psd = torch.fft.fft2(dist)
        return R_z, torch.sqrt(psd)
    
    def _tilt_mat(self):
        '''
        This function generates the correlation matrix for zernike coefficients associated with
            x and y tilts for the entire image
        '''
        Nh, Nw = self.gh, self.gw
        delta0 = self.turb_params['L']*self.turb_params['wavelength']/(2*self.D) * self.turb_params['adj']
        c1 = 2*((24/5)*math.gamma(6/5))**(5/6)
        c2 = 4*c1/math.pi*(math.gamma(11/6))**2
        smax = delta0/self.D*max(Nh, Nw)
        spacing = delta0/self.D
        # I0_I2: precomputed integration of the Bessel function of the first kind (order 0 and 2)
        I0_0 = self.I0_I2['I0_0'] # order 0, input 0
        
        I0_arr, I2_arr = self._sample_in_vec(smax, spacing)
        h, w = torch.meshgrid(torch.arange(1, Nh+1), torch.arange(1, Nw+1))
        s = torch.sqrt((h-Nh/2)**2 + (w-Nw/2)**2).cuda()

        C0 = (I0_arr[s.long()] + I2_arr[s.long()])/I0_0
        C0[int(Nh/2), int(Nw/2)] = 1
        C0_scaled = C0 * I0_0 * c2 * 2 * math.pi * \
                (self.Dr0/2)**(5/3) * (2*self.turb_params['wavelength']/(math.pi*self.D)) ** 2
        Cfft = torch.fft.fft2(C0_scaled)
        S_half = torch.sqrt(Cfft)
        S_half[S_half.abs() < self.turb_params['thre']*S_half.abs().max()] = 0
        S_half_new = torch.stack((S_half.real, S_half.imag), dim=0)
        const = math.sqrt(2) * max(Nh, Nw) * (self.turb_params['L']/delta0)
        return const, S_half_new
        
    def _sample_in_vec(self, smax, spacing):
        s_arr = torch.arange(0, smax, spacing)
        sample_idx = torch.argmin(torch.abs(s_arr.unsqueeze(1) - self.I0_I2['s'].unsqueeze(0)), dim=1)
        return self.I0_I2['i0_integral'][sample_idx].float().cuda(), \
                self.I0_I2['i2_integral'][sample_idx].float().cuda()
        
    def forward(self, img, noise = None):
        b, c = img.shape[0:2]
        
        if noise is None: 
            noise_zer = torch.randn((b, 36, self.gh, self.gw), device=self.device)
            noise_tilt = torch.randn(b, self.gh, self.gw, 2, device=self.device)
        else: 
            noise_zer = noise[0]
            noise_tilt = noise[1]
        
        noise = (noise_zer, noise_tilt)
        
        if not img.shape[2:] == (self.h, self.w):
            self.h, self.w = img.shape[2:]
            print('new input image shape:', self.h, self.w)
            if self.h > self.gh:
                self.gh = self.h
            if self.w > self.gw:
                self.gw = self.w
            self.turb_params['img_size'] = img.shape[2:-1]
            self.const, self.S_half = self._tilt_mat()
            self.R_z, self.sqrt_psd = self._corr_mat(self.turb_params['corr'])
            yy, xx = torch.meshgrid(torch.arange(0, self.h),torch.arange(0, self.w))
            self.pixel_pos = torch.stack((xx, yy), -1).unsqueeze(0).type(torch.float32).cuda()
        # convolution of psf basis and the input image
        img_pad = F.pad(img.view((-1, 1, self.h, self.w)), (16, 16, 16, 16), mode='reflect')
        img_mean = F.conv2d(img_pad, self.mu).view(b, -1, self.h, self.w)
        dict_img = F.conv2d(img_pad, self.basis_psf).view(b, c, -1, self.h, self.w)
        # generate an random field of zernike coefficients
        random_ = torch.sqrt(self.Dr0 ** (5 / 3))*noise_zer
        zc = torch.einsum('ij,kj...->ki...', self.R_z, random_) # 4,36,16,16
        noise_spec = torch.fft.fft2(zc, dim=(2,3))
        zer = torch.fft.ifft2(self.sqrt_psd.unsqueeze(0).unsqueeze(0)*noise_spec, dim=(-2,-1)).real
        zer = zer[:, :, :self.h, :self.w]
        # apply P2S to get random field of the psf coefficients
        weight = self.mapping(zer.permute(0, 2, 3, 1).reshape(b, self.h*self.w, -1))
        weight = weight.view((b, self.h, self.w, 100)).permute(0, 3, 1, 2)  # target: (100,512,512)
        # get blurred image
        out_blur = torch.sum(weight.unsqueeze(1)*dict_img, 2) + img_mean
        # distort the blurred image with tilt maps
        pos_shift = torch.fft.irfft2((self.S_half.permute(1, 2, 0).unsqueeze(0) * noise_tilt), s=(self.gh, self.gw), dim=(1,2)) * self.const
        pos_shift = pos_shift[:, :self.h, :self.w, :]
        flow = 2.0*(self.pixel_pos + pos_shift) / (torch.tensor((self.w, self.h))-1).cuda() - 1.0
        out = F.grid_sample(out_blur, flow, 'bilinear', padding_mode='border', align_corners=False)
        return noise, flow, out_blur, out


class _P2S(nn.Module):
    def __init__(self, input_dim=36, output_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, output_dim)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc2(y))
        out = self.out(y)
        return out
        

def _nollCovMat(Z, D, fried):
    # correlation matrix for all zernike coefficients
    C = torch.zeros((Z, Z))
    x = torch.linspace(0, Z-1, Z)
    xv, yv = torch.meshgrid(x, x)

    n2i = torch.tensor([_nollToZernInd(i+1) for i in range(Z)])
    n = n2i[:, 0].squeeze()
    m = n2i[:, 1].squeeze()
    ni, nj = torch.meshgrid(n, n)
    mi, mj = torch.meshgrid(m, m)

    mask = (torch.abs(mi) == torch.abs(mj)) * (torch.remainder(xv - yv, 2) == 0)
    num = torch.lgamma(torch.tensor(14.0/3.0)) + torch.lgamma((ni + nj - 5.0/3.0)/2.0)
    den = torch.lgamma((-ni + nj + 17.0/3.0)/2.0) + torch.lgamma((ni - nj + 17.0/3.0)/2.0) + \
        torch.lgamma((ni + nj + 23.0/3.0)/2.0)
    coef1 = 0.0072 * (np.pi ** (8.0/3.0)) * ((D/fried) ** (5.0/3.0)) * torch.sqrt((ni + 1) * (nj + 1)) * \
        ((-1) ** ((ni + nj - 2*torch.abs(mi))/2.0 * mask))
    C = coef1 * torch.exp(num)/torch.exp(den) * mask
    C[0, 0] = 1
    return C

def _nollToZernInd(j):
    """
    Authors: Tim van Werkhoven, Jason Saredy
    See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1) % 2)) / 2.0))
    return n, m
