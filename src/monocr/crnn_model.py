#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crnn model architecture for mon ocr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
from typing import List

class CRNN(nn.Module):
    """crnn model for mon ocr"""
    
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # cnn architecture
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
        # lstm layers
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "CNN height must be 1"
        conv = conv.squeeze(2).permute(0, 2, 1)  # [B, W, C]
        
        # lstm layers
        recurrent, _ = self.lstm1(conv)
        recurrent, _ = self.lstm2(recurrent)
        
        # dropout and final classification
        recurrent = self.dropout(recurrent)
        out = self.fc(recurrent)
        return out  # [B, W, num_classes]


def build_charset(corpus_dir: str) -> str:
    """build charset from corpus files"""
    charset = set()
    
    # search for text files in corpus directory
    for ext in ['*.txt']:
        pattern = os.path.join(corpus_dir, '**', ext)
        for file_path in glob.glob(pattern, recursive=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    charset.update(content)
            except:
                continue
    
    # remove whitespace and control characters
    charset = {c for c in charset if c.strip() and ord(c) >= 32}
    
    # sort for consistent ordering
    charset_str = ''.join(sorted(charset))
    return charset_str