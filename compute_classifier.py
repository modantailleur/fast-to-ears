import pandas as pd
import torch
from tqdm import tqdm
from transcoder.transcoders import ThirdOctaveToMelTranscoderPinv, ThirdOctaveToMelTranscoder
import pickle
import numpy as np
import argparse
import os
import h5py

class DatasetGenerator(object):
    def __init__(self, spectral_data_path):
        _, self.extension = os.path.splitext(spectral_data_path)
        self.cense_data_path = spectral_data_path
        if self.extension == '.h5':
            with h5py.File(self.cense_data_path, 'r') as hf:
                self.spectral_data = hf['fast_125ms'][:]
        elif self.extension == '.npy':
            self.spectral_data = np.load(spectral_data_path)

        self.len_dataset = len(self.spectral_data)

    def __getitem__(self, idx):
        spectral_data = self.spectral_data[idx]
        return spectral_data

    def __len__(self):
        return self.len_dataset

#for CNN + PINV
class TranscoderPANNEvaluater:
    def __init__(self, transcoder, eval_dataset, dtype=torch.FloatTensor, db_offset=-88):
        self.dtype = dtype
        self.transcoder = transcoder
        self.eval_dataset = eval_dataset
        self.db_offset = db_offset

    def evaluate(self, batch_size=32, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        eval_outputs = np.array([])

        for (spectral_data) in tqdm_it:
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)
            
            spectral_data = spectral_data + self.db_offset

            #If the third-octaves are less than 10s in length, then the entire third-octave bins are send to PANN as input.
            #If not, then only chunks of 10s frames are send to PANN as input. This is because PANN is most performant with 
            #10s inputs, as it was trained on 10s audio chunks. 
            if spectral_data.shape[1] < 80:
                _ , presence = self.transcoder.thirdo_to_mels_to_logit(spectral_data, frame_duration=spectral_data.shape[1]//8)
            else:
                _ , presence = self.transcoder.thirdo_to_mels_to_logit(spectral_data, frame_duration=10)
            presence = torch.mean(presence, axis=-1)
            if len(eval_outputs) != 0:
                eval_outputs = torch.cat((eval_outputs, presence), dim=0)
            else:
                eval_outputs = presence
        eval_outputs = eval_outputs.detach().cpu().numpy()
        return(eval_outputs)

class LevelEvaluater:
    def __init__(self, eval_dataset, dtype=torch.FloatTensor):
        self.dtype = dtype
        self.eval_dataset = eval_dataset
        self.fn=np.array([20, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])

    def evaluate(self, batch_size=1, device=torch.device("cpu")):
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        eval_outputs = np.array([])

        for (spectral_data) in tqdm_it:
            spectral_data = spectral_data.type(self.dtype)
            spectral_data = spectral_data.to(device)

            puiss_spectral_data = 10**(spectral_data/10)
            sum_puiss_spectral_data = torch.sum(puiss_spectral_data, axis=-1)
            level_spectral_data = 10*torch.log10(sum_puiss_spectral_data)
            level_spectral_data = level_spectral_data.view(-1)
            
            if len(eval_outputs) != 0:
                eval_outputs = torch.cat((eval_outputs, level_spectral_data), dim=0)
            else:
                eval_outputs = level_spectral_data
                        
        eval_outputs = eval_outputs.detach().cpu().numpy()
        return(eval_outputs)

def calculate_db_offset(input_path='./spectral_data/', spectral_data_name= 'test.npy'):
    #transcoder setup
    MODEL_PATH = "./reference_models"
    cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    transcoder = 'cnn_pinv'
    dtype=torch.FloatTensor
    fs=32000
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

    spectral_path = input_path + spectral_data_name
    dataset = DatasetGenerator(spectral_data_path=spectral_path)

    evaluater = LevelEvaluater(eval_dataset=dataset)
    batch_size = 1

    eval_outputs = evaluater.evaluate(batch_size=batch_size, device=device)

    eval_outputs = eval_outputs.reshape(-1)
    db_offset =  - np.percentile(eval_outputs, 99)

    print('DB OFFSET')
    print(db_offset)

    return(db_offset)

def compute_predictions(db_offset, batch_size=1, input_path='./spectral_data/', output_path='./predictions/', spectral_data_name= 'test.npy', output_format='h5'):

    #transcoder setup
    MODEL_PATH = "./reference_models"
    cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    transcoder = 'cnn_pinv'
    dtype=torch.FloatTensor
    fs=32000
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

    batch_size = 1

    spectral_path = input_path + spectral_data_name
    dataset = DatasetGenerator(spectral_data_path=spectral_path)

    transcoder_cnn_logits_pann = ThirdOctaveToMelTranscoder(transcoder, cnn_logits_name, MODEL_PATH, device=device)
    evaluater = TranscoderPANNEvaluater(transcoder=transcoder_cnn_logits_pann, eval_dataset=dataset, db_offset=db_offset, dtype=dtype)

    eval_outputs = evaluater.evaluate(batch_size=batch_size, device=device)

    f_name, _ = os.path.splitext(spectral_data_name)

    if output_format == "npy":
        np.save(output_path + 'predictions_' + f_name + '.npy', eval_outputs)
    elif output_format == "h5":
        with h5py.File(output_path + 'predictions_' + f_name + '.h5', "w") as h5_file:
            # Create a dataset in the HDF5 file
            dataset = h5_file.create_dataset("predictions", data=eval_outputs)

def main(config):
    batch_size=1
    input_path=config.input_path
    output_path=config.output_path
    spectral_data_name= config.spectral_data_name
    output_format = config.output_format

    if config.get_db_offset:
        db_offset = calculate_db_offset(input_path=input_path, spectral_data_name= spectral_data_name)
    else:
        db_offset = config.db_offset
    
    compute_predictions(db_offset=db_offset, batch_size=batch_size, input_path=input_path, output_path=output_path, spectral_data_name=spectral_data_name, output_format=output_format)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')
    parser.add_argument('-i', '--input_path', type=str, default='./spectral_data/',
                        help='The path where the third-octave data files are stored, in a npy format')
    parser.add_argument('-o', '--output_path', type=str, default='./predictions/',
                        help='The path where to store the predictions')
    parser.add_argument('-of', '--output_format', type=str, default='h5',
                        help='The output format to used for the predictions (h5 or npy)')
    parser.add_argument('-n', '--spectral_data_name', type=str, default='test.npy',
                        help='name of the spectral data file in npy or h5 format')
    parser.add_argument('-dbo', '--db_offset', type=float, default=-90,
                        help='dB offset to apply to the measured third octaves. Needs to be calculated beforehand.')
    parser.add_argument('-gdbo', '--get_db_offset', type=bool, default=False,
                        help='If set to True, only calculates the dB offset based on the max value of the given dataset.')
    config = parser.parse_args()
    main(config)



