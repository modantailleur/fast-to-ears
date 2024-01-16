#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:45:28 2022

@author: user
"""

import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import copy
from sklearn.model_selection import KFold
from dataloader import Dataset
import models as md

#for CNN + PINV
class DLModelsTrainer:
    """
    Trainer to train the coarser (527 --> 3)
    """
    def __init__(self, models_path, scores, groundtruth, fname, scores_len, learning_rate=1e-3, dtype=torch.FloatTensor, 
                 ltype=torch.LongTensor):
        
        self.dtype = dtype
        self.ltype = ltype
        
        self.models_path = models_path

        k_folds = 10
        self.kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        self.dataset_fold = self.kfold.split(fname)

        self.scores_len = scores_len
        self.model = md.FC(scores_len)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.model_fold = []

        self.scores = scores
        self.groundtruth = groundtruth
        self.fname = fname
        self.train_fold = []
        self.eval_fold = []

    def train(self, batch_size=128, epochs=100, device=torch.device("cpu")):
        losses_train_fold = []
        losses_eval_fold = []

        for fold, (train_id, eval_id) in enumerate(self.dataset_fold):  
            self.train_fold.append(train_id)
            self.eval_fold.append(eval_id)

            train_scores = self.scores[train_id]
            train_groundtruth = self.groundtruth[train_id]
            eval_scores = self.scores[eval_id]
            eval_groundtruth = self.groundtruth[eval_id]

            model = copy.deepcopy(self.model)
            optimizer = optim.Adam(params=model.parameters(), lr=self.learning_rate)
            loss_function = nn.BCELoss()
            model = model.to(device)

            cur_loss = 0

            train_dataset = Dataset(train_scores, train_groundtruth)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)

            losses_train = []
            losses_eval = []
            #train
            for epoch in range(epochs):
                tqdm_it=tqdm(train_dataloader, 
                            desc='TRAINING: Epoch {}, loss: {:.4f}'
                            .format(epoch+1, cur_loss))
                for (x,y) in tqdm_it:
                    x = x.type(self.dtype)
                    y = y.type(self.dtype)
                    
                    x = x.to(device)
                    y = y.to(device)
                    
                    optimizer.zero_grad()

                    y_pred = model(x)

                    loss = loss_function(y_pred,y)

                    loss.backward()
                    
                    optimizer.step()
                    
                    cur_loss = float(loss.data)

                    losses_train.append(cur_loss)
                    
                    tqdm_it.set_description('TRAINING: Epoch {}, loss: {:.4f}'
                                            .format(epoch+1,cur_loss))                    

                loss_eval = self.validate(eval_scores, eval_groundtruth, model, epoch+1, batch_size=batch_size, device=device, label='EVALUATION')
                losses_eval.append(loss_eval)
            
            losses_train_fold.append(losses_train)
            losses_eval_fold.append(losses_eval)
            self.model_fold.append(model.state_dict())

        return(losses_train_fold, losses_eval_fold)

    def validate(self, eval_scores, eval_groundtruth, model, currentEpoch, batch_size=64, device=torch.device("cpu"), label='VALIDATION', forced=False,
                    n_chunk=100):

        loss_valid = 0
        len_valid_dataset = 0
        loss_function =  nn.BCELoss(reduction='none')
        all_losses = torch.Tensor().to(device)

        valid_dataset = Dataset(eval_scores, eval_groundtruth)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(valid_dataloader, desc=label+':loss {:.4f}'.format(0))
        for (x,y) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            loss = loss_function(y_pred,y).mean(dim=[1]).detach()
            all_losses = torch.cat((all_losses,loss))

        loss_valid = torch.mean(all_losses)
        tqdm_it.set_description(label+':loss {:.4f}'.format(loss_valid))  
        print(f'loss_valid:{loss_valid}')
        return loss_valid.detach().cpu().numpy()

