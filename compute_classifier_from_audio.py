
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:02:54 2022

@author: user
"""
import numpy as np
import librosa
import numpy as np
import librosa
import torch.utils.data
import torch
from transcoder.transcoders import ThirdOctaveToMelTranscoderPinv, ThirdOctaveToMelTranscoder
from utils.util import sort_labels_by_score
import argparse

def main(config):
    MODEL_PATH = "./reference_models"
    filename = config.audio_file
    cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    cnn_logits_slow_name = 'classifier=PANN+dataset=full+dilation=1+epoch=50+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+tho_type=slow+transcoder=cnn_pinv+ts=1_model'
    cnn_mels_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+step=train+transcoder=cnn_pinv+ts=0_model'
    transcoder = 'cnn_pinv'
    fs=32000
    full_filename = "audio/" + filename
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
        #MT: add
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        #MT: add
        device = torch.device("cpu")

    transcoder_deep_bce = ThirdOctaveToMelTranscoder(transcoder, cnn_logits_name, MODEL_PATH, device=device)
    # transcoder_deep_mse = ThirdOctaveToMelTranscoder(transcoder, cnn_mels_name, MODEL_PATH, device=device)
    # transcoder_pinv = ThirdOctaveToMelTranscoderPinv(MODEL_PATH, cnn_logits_name, device, classifier="PANN")

    x_32k = librosa.load(full_filename, sr=fs)[0]
    # x_32k = librosa.util.normalize(x_32k)

    #Groundtruth mels
    _, x_logit_gt = transcoder_deep_bce.wave_to_mels_to_logit(x_32k)

    #PANN-1/3oct model
    x_mels_deep_bce = transcoder_deep_bce.wave_to_thirdo_to_mels(x_32k)
    x_logit_deep_bce = transcoder_deep_bce.mels_to_logit(x_mels_deep_bce, mean=True)
    x_logit_deep_bce = x_logit_deep_bce.T

    #CNN-mels model
    # x_mels_deep_mse = transcoder_deep_bce.wave_to_thirdo_to_mels(x_32k)
    # x_logit_deep_mse = transcoder_deep_mse.mels_to_logit(x_mels_deep_mse, mean=True)
    # x_logit_deep_mse = x_logit_deep_mse.T

    print('\n XXXXXXXXX PANN-Mels (mel input) XXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_gt, axis=0), transcoder_deep_bce.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_gt, axis=0), transcoder_deep_bce.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')

    print('\n XXXXXXXXXXXX PANN-1/3oct (fast third-octave input) XXXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_deep_bce, axis=0), transcoder_deep_bce.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_deep_bce, axis=0), transcoder_deep_bce.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get PANN-Mels and PANN-1/3oct predictions of a given audio file')

    parser.add_argument('audio_file', type=str,
                        help='Name of the original audio file that should be located in the "audio" folder')

    config = parser.parse_args()
    main(config)