#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:51:42 2019

@author: ceyx
"""

from torch.utils.data import Dataset

class CF_Dataset(Dataset):
    
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()
    
    def __len__(self):
        return len(self.data)
    