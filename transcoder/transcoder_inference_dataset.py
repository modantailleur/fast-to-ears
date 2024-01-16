#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:05:35 2022

@author: user
"""
import os
import sys
import transcoder.models_transcoder as md
from transcoder.transcoders import ThirdOctaveToMelTranscoder, ThirdOctaveToMelTranscoderPinv

# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

import librosa
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os
import numpy as np
import librosa
import numpy as np
import librosa
import torch.utils.data
import torch
import utils.util as ut

random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    # Set the random seed for GPU (if available)
    torch.cuda.manual_seed(0)

class YamNetInference():
    def __init__(self, device=torch.device("cpu"), verbose=True, normalize=True, db_offset=0):
        self.n_labels = 521

        model_path = "./reference_models"
        cnn_yamnet_name = 'classifier=YamNet+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
        model_type = 'cnn_pinv'
        self.dtype=torch.FloatTensor

        self.transcoder_deep_yamnet = ThirdOctaveToMelTranscoder(model_type, cnn_yamnet_name, model_path, device)
        self.normalize = normalize
        self.verbose = verbose
        self.db_offset = db_offset
        self.db_offset_multiplier = 10**(db_offset/10)

    def inference_from_scratch(self, file_name, mean=True):
        x_16k = librosa.load(file_name, sr=16000)[0]
        if (self.db_offset is not None) and (self.normalize == False):
            x_16k = x_16k * self.db_offset_multiplier
        if self.normalize:
            x_16k = librosa.util.normalize(x_16k)

        #x_mels_yamnet_cnn, x_logit_yamnet_cnn = self.transcoder_deep_yamnet.transcode_from_wav(x_16k, self.dtype)
        x_mels_yamnet_gt = self.transcoder_deep_yamnet.mels_tr.wave_to_mels(x_16k)
        x_logit_yamnet_gt = self.transcoder_deep_yamnet.mels_to_logit(x_mels_yamnet_gt)

        if mean:
            x_logit_yamnet_gt = np.mean(x_logit_yamnet_gt, axis=0)
            x_logit_yamnet_gt = np.expand_dims(x_logit_yamnet_gt, axis=0)

        if self.verbose:
            print('\n XXXXXXXXXXXX YAMNET CLASSIFIER (MEL INPUT) XXXXXXXXXXXX')
            print(file_name)
            labels = ut.sort_labels_by_score(np.mean(x_logit_yamnet_gt, axis=0), self.transcoder_deep_yamnet.classif_inference.labels_str)[1][:10]
            scores = ut.sort_labels_by_score(np.mean(x_logit_yamnet_gt, axis=0), self.transcoder_deep_yamnet.classif_inference.labels_str)[0][:10]
            for k in range(len(labels)):
                print(f'{labels[k]} : {round(float(scores[k]), 2)}')

        return(x_logit_yamnet_gt)

class TrYamNetInference():
    def __init__(self, device=torch.device("cpu"), verbose=False, normalize=True):
        self.n_labels = 521

        model_path = "./reference_models"
        cnn_yamnet_name = 'classifier=YamNet+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
        model_type = 'cnn_pinv'

        self.transcoder_deep_yamnet = ThirdOctaveToMelTranscoder(model_type, cnn_yamnet_name, model_path, device)
        self.normalize = normalize

    def inference_from_scratch(self, file_name, mean=True):
        x_16k = librosa.load(file_name, sr=16000)[0]
        if self.normalize:
            x_16k = librosa.util.normalize(x_16k)

        x_mels_yamnet_cnn = self.transcoder_deep_yamnet.wave_to_thirdo_to_mels(x_16k)
        x_logit_yamnet_cnn = self.transcoder_deep_pann.mels_to_logit(x_mels_yamnet_cnn, mean=mean)
        x_logit_yamnet_cnn = np.mean(x_logit_yamnet_cnn, axis=1)
        x_logit_yamnet_cnn = x_logit_yamnet_cnn.reshape(1, -1)

        return(x_logit_yamnet_cnn)

class PANNInference():
    def __init__(self, device=torch.device("cpu"), verbose=False, constant_10s_audio=False, normalize=True, db_offset=0, pann_type='ResNet38'):
        self.n_labels = 527

        model_path = "./reference_models"
        cnn_pann_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
        model_type = 'cnn_pinv'
        self.dtype=torch.FloatTensor

        self.verbose = verbose
        self.transcoder_deep_pann = ThirdOctaveToMelTranscoder(model_type, cnn_pann_name, model_path, device, pann_type=pann_type)
        self.constant_10s_audio = constant_10s_audio
        self.normalize = normalize
        self.db_offset = db_offset
        self.db_offset_multiplier = 10**(db_offset/10)

    def inference_from_scratch(self, file_name, mean=True, to_tvb=False):
        x_32k = librosa.load(file_name, sr=32000)[0]
        if (self.db_offset is not None) and (self.normalize == False):
            x_32k = x_32k * self.db_offset_multiplier
        if self.normalize:
            x_32k = librosa.util.normalize(x_32k)

        #if audio is exactly 10s long, speeds up the process
        if self.constant_10s_audio:
            x_mels_pann_gt = self.transcoder_deep_pann.mels_tr.wave_to_mels(x_32k)
            x_logit_pann_gt = self.transcoder_deep_pann.mels_to_logit(x_mels_pann_gt, mean=mean)
            # x_logit_pann_gt = x_logit_pann_gt[0]
            if not mean:
                x_logit_pann_gt = x_logit_pann_gt.reshape(x_logit_pann_gt.shape[0]*x_logit_pann_gt.shape[1]*x_logit_pann_gt.shape[2], x_logit_pann_gt.shape[-1])
        else:
            # x_mels_pann_gt = self.transcoder_deep_pann.mels_tr.wave_to_mels(x_32k)
            # x_logit_pann_gt = self.transcoder_deep_pann.mels_to_logit_sliced(x_mels_pann_gt, mean=mean)

            # x_mels_pann_gt = self.transcoder_deep_pann.wave_to_mels_sliced(x_32k, frame_duration=10)
            # x_logit_pann_gt = self.transcoder_deep_pann.mels_to_logit_sliced(x_mels_pann_gt, mean=mean)
            
            x_mels_pann_gt, x_logit_pann_gt = self.transcoder_deep_pann.wave_to_mels_to_logit(x_32k, mean=mean)
            # x_logit_pann_gt = x_logit_pann_gt.reshape(1, -1)

        if self.verbose:
            print('\n XXXXXXXXXXXX PANN CLASSIFIER (MEL INPUT) XXXXXXXXXXXX')
            print(file_name)
            labels = ut.sort_labels_by_score(np.mean(x_logit_pann_gt, axis=0), self.transcoder_deep_pann.classif_inference.labels_str)[1][:10]
            scores = ut.sort_labels_by_score(np.mean(x_logit_pann_gt, axis=0), self.transcoder_deep_pann.classif_inference.labels_str)[0][:10]
            for k in range(len(labels)):
                print(f'{labels[k]} : {round(float(scores[k]), 2)}')

        if to_tvb:
            x_logit_pann_gt = self.transcoder_deep_pann.classif_inference.batch_logit_to_tvb(x_logit_pann_gt)

        return(x_logit_pann_gt)

class TrPANNInference():
    def __init__(self, device=torch.device("cpu"), verbose=False, constant_10s_audio=False, normalize=True, db_offset=0, pann_type='ResNet38'):
        self.n_labels = 527

        model_path = "./reference_models"
        cnn_pann_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
        model_type = 'cnn_pinv'
        self.dtype=torch.FloatTensor

        self.transcoder_deep_pann = ThirdOctaveToMelTranscoder(model_type, cnn_pann_name, model_path, device, pann_type=pann_type)
        self.constant_10s_audio = constant_10s_audio
        self.normalize = normalize

        self.verbose = verbose
        self.db_offset = db_offset
        self.db_offset_multiplier = 10**(db_offset/20)

    def inference_from_scratch(self, file_name, mean=True, to_tvb=False):
        x_32k = librosa.load(file_name, sr=32000)[0]
        if (self.db_offset is not None) and (self.normalize == False):
            x_32k = x_32k * self.db_offset_multiplier

        if self.normalize:
            x_32k = librosa.util.normalize(x_32k)

        if self.constant_10s_audio:
            x_mels_pann_cnn = self.transcoder_deep_pann.wave_to_thirdo_to_mels(x_32k)
            x_logit_pann_cnn = self.transcoder_deep_pann.mels_to_logit(x_mels_pann_cnn, mean=mean)
            # x_mels_pann_cnn, x_logit_pann_cnn = self.transcoder_deep_pann.wave_to_thirdo_to_logits(x_mels_pann_cnn, mean=mean)
            if not mean:
                x_logit_pann_cnn = x_logit_pann_cnn.reshape(x_logit_pann_cnn.shape[0]*x_logit_pann_cnn.shape[1]*x_logit_pann_cnn.shape[2], x_logit_pann_cnn.shape[-1])

        else:
            x_mels_pann_cnn = self.transcoder_deep_pann.wave_to_thirdo_to_mels(x_32k)
            x_logit_pann_cnn = self.transcoder_deep_pann.mels_to_logit(x_mels_pann_cnn, mean=mean)
            x_logit_pann_cnn = x_logit_pann_cnn.T

            # if (self.db_offset is not None) and (self.normalize == False):
            #     x_mels_pann_cnn = x_mels_pann_cnn + self.db_offset

            #MT: removed
            # x_logit_pann_cnn = np.mean(x_logit_pann_cnn, axis=1)
            # x_logit_pann_cnn = x_logit_pann_cnn.reshape(1, -1)

        if self.verbose:
            print('\n XXXXXXXXXXXX TRANSCODED PANN CLASSIFIER (THIRD-OCTAVE INPUT) USING CNN-LOGITS TRANSCODER XXXXXXXXXXXX')
            print(file_name)
            labels = ut.sort_labels_by_score(np.mean(x_logit_pann_cnn, axis=0), self.transcoder_deep_pann.classif_inference.labels_str)[1][:10]
            scores = ut.sort_labels_by_score(np.mean(x_logit_pann_cnn, axis=0), self.transcoder_deep_pann.classif_inference.labels_str)[0][:10]
            for k in range(len(labels)):
                print(f'{labels[k]} : {round(float(scores[k]), 2)}')

        if to_tvb:
            x_logit_pann_cnn = self.transcoder_deep_pann.classif_inference.batch_logit_to_tvb(x_logit_pann_cnn)

        return(x_logit_pann_cnn)

class TrPANNInferenceSlow():
    def __init__(self, device=torch.device("cpu"), verbose=False, constant_10s_audio=False, normalize=True, db_offset=0):
        self.n_labels = 527

        model_path = "./reference_models"
        cnn_pann_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
        model_type = 'cnn_pinv'
        self.dtype=torch.FloatTensor

        self.transcoder_deep_pann = ThirdOctaveToMelTranscoder(model_type, cnn_pann_name, model_path, device, flen=32758, hlen=32000)
        self.constant_10s_audio = constant_10s_audio
        self.normalize = normalize

        self.verbose = verbose
        self.db_offset = db_offset
        self.db_offset_multiplier = 10**(db_offset/10)

    def inference_from_scratch(self, file_name):
        x_32k = librosa.load(file_name, sr=32000)[0]
        if (self.db_offset is not None) and (self.normalize == False):
            x_32k = x_32k * self.db_offset_multiplier

        if self.normalize:
            x_32k = librosa.util.normalize(x_32k)

        if self.constant_10s_audio:
            x_mels_pann_cnn = self.transcoder_deep_pann.transcode_from_wav_entire_file(x_32k)
            # if (self.db_offset is not None) and (self.normalize == False):
            #     x_mels_pann_cnn = x_mels_pann_cnn + self.db_offset
            x_logit_pann_cnn = self.transcoder_deep_pann.mels_to_logit(x_mels_pann_cnn, slice=False)
            x_logit_pann_cnn = x_logit_pann_cnn.reshape(1, -1)
        else:
            x_mels_pann_cnn, x_logit_pann_cnn = self.transcoder_deep_pann.transcode_from_wav(x_32k, frame_duration=10)
            # if (self.db_offset is not None) and (self.normalize == False):
            #     x_mels_pann_cnn = x_mels_pann_cnn + self.db_offset
            x_logit_pann_cnn = np.mean(x_logit_pann_cnn, axis=1)
            x_logit_pann_cnn = x_logit_pann_cnn.reshape(1, -1)

        if self.verbose:
            print('\n XXXXXXXXXXXX TRANSCODED PANN CLASSIFIER (THIRD-OCTAVE INPUT) USING CNN-LOGITS TRANSCODER XXXXXXXXXXXX')
            print(file_name)
            labels = ut.sort_labels_by_score(x_logit_pann_cnn[0], self.transcoder_deep_pann.classif_inference.labels_str)[1][:10]
            scores = ut.sort_labels_by_score(x_logit_pann_cnn[0], self.transcoder_deep_pann.classif_inference.labels_str)[0][:10]
            for k in range(len(labels)):
                print(f'{labels[k]} : {round(float(scores[k]), 2)}')

        return(x_logit_pann_cnn)

