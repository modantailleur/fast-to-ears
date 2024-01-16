#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:07:32 2022

@author: user
"""
import torch
import torch.utils.data
import torch.nn.functional as F
import librosa 

def pinv(x, tho_tr, mels_tr, reshape=None, dtype=torch.FloatTensor, device=torch.device("cpu"), input_is_db=True):
    """Convert a third octave spectrogram to a mel spectrogram using 
    a pseudo-inverse method.

    Parameters
    ----------
    x : torch.Tensor
        input third octave spectrogram of size (batch size, third octave 
                                                transform time bins, 
                                                third octave transform
                                                frequency bins)
        
    tho_tr : ThirdOctaveTransform instance
        third octave transform used as input (see ThirdOctaveTransform class)
    
    mels_tr : mels transform classes instance
        mels bands transform to match (see PANNMelsTransform for example)
    
    reshape : int
        if not set to None, will reshape the input tensor to match the given
        reshape value in terms of time bins. Simple copy of every time bin
        with some left and right extensions if 'reshape' is not a power of 
        two of the original 'time bins' value from the input tensor. 
        
    dtype : 
        data type to apply
        

    Returns
    -------
    x_mels_pinv : torch.Tensor
        mel spectrogram of size (batch size, mel transform time bins, 
                                 mel transform frequency bins)
    """

    #x_phi_inv: (2049,29)
    x_phi_inv = tho_tr.inv_tho_basis_torch

    #remove the log component from the input third octave spectrogram
    x_power = tho_tr.db_to_power_torch(x, device=device)

    # The energy is not the same depending on the size of the temporal 
    # window that is used. The larger the window, the bigger the energy 
    # (because it sums on every frequency bin). A scale factor is needed
    # in order to correct this. For example, with a window size of 1024, 
    # the number of frequency bins after the fft will be 513 (N/2 +1). 
    # With a window size of 4096, the number of frequency bins after the
    # fft will be 2049 (N/2 +1). The scaling factor is then 2049/513.

    scaling_factor = (1+mels_tr.flen/2) / (1+tho_tr.flen/2)
    if mels_tr.window == 'hann':
        #hann window loses 50% of energy
        scaling_factor = scaling_factor * 0.5
    else:
        raise Exception("Window unrecognised.") 

    # scaling factor is raised to the power of 2 because it is supposed de be used 
    # on an energy spectrum and x_power is a power spectrum
    scaling_factor = scaling_factor ** 2
    
    # scaling of the power spectrum to fit the scale of the stft used in the Mel transform
    x_power = x_power * scaling_factor

    #put tensor to correct device
    x_phi_inv.to(x.dtype)
    x_power.to(x.dtype)

    x_phi_inv = x_phi_inv.to(device)
    x_power = x_power.to(device)

    #add one dimension to the pseudo-inverted matrix 
    #for the batch size of the input
    x_phi_inv = x_phi_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)

    #permute third octave transform time bins and third octave transform 
    # frequency bins to allow matrix multiplication
    x_power = torch.permute(x_power, (0,2,1))

    if tho_tr.tho_freq:
        x_spec_pinv = torch.matmul(x_phi_inv, x_power)
        
    else:
        x_mel_inv = librosa.filters.mel(sr=32000, n_fft=4096, fmin=50, fmax=14000, n_mels=64)
        x_mel_inv = x_mel_inv.T
        x_mel_inv = torch.from_numpy(x_mel_inv)
        x_mel_inv = x_mel_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)
        x_spec_pinv = torch.matmul(x_mel_inv, x_power)
        #x_spec_pinv = x_power

    #eventually reshape time dimension to match 'reshape' value
    if reshape:
        x_spec_pinv = F.interpolate(x_spec_pinv, size=reshape, scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)
        #permute again to have the dimensions in correct order 
        x_spec_pinv = torch.permute(x_spec_pinv, (0,2,1))

    else: 
        x_spec_pinv = x_spec_pinv

    if input_is_db:
        #from power spectrogram to mel spectrogram
        x_mels_pinv = mels_tr.power_to_mels(x_spec_pinv)
    else:
        if mels_tr.name == "yamnet":
            raise Exception("It is not possible to train regular Mel spectrogram for YamNet (as opposed to logMel Spectrogram") 
        if mels_tr.name == "pann":
            x_mels_pinv = mels_tr.power_to_mels_no_db(x_spec_pinv)
    
    x_mels_pinv = x_mels_pinv.squeeze(0)

    return(x_mels_pinv)
