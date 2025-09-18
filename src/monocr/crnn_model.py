#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNN Model Architecture for Mon OCR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
from typing import List

class CRNN(nn.Module):
    """CRNN model for Mon OCR - matches the trained model architecture"""
    
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # Enhanced CNN architecture for better capacity
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64->32
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32->16
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 16->8
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8->4
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (4, 1), 1, 0),  # 4->1
            nn.ReLU(),
        )
        # Two LSTM layers for better sequence modeling
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)  # add dropout to prevent overfitting
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "CNN height must be 1"
        conv = conv.squeeze(2).permute(0, 2, 1)  # [B, W, C]
        
        # Two LSTM layers for better sequence modeling
        recurrent, _ = self.lstm1(conv)
        recurrent, _ = self.lstm2(recurrent)
        
        # Apply dropout before final classification
        recurrent = self.dropout(recurrent)
        out = self.fc(recurrent)
        return out  # [B, W, num_classes]

def build_charset(corpus_dir: str) -> str:
    """Build charset from corpus files"""
    charset = set()
    txt_files = glob.glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True)
    
    for fpath in txt_files:
        if os.path.getsize(fpath) == 0:
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    charset.update(line.strip())
        except Exception:
            continue
    
    charset_str = "".join(sorted(list(charset)))
    return charset_str
