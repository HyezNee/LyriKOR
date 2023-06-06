import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from transformers import BartForConditionalGeneration
from kobart import get_pytorch_kobart_model
import numpy as np

import logging

from kobart import get_pytorch_kobart_model


class CustomKoBART(nn.Module):
    def __init__(self):
        super(CustomKoBART, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self.model.to(self.device)
        self.model.train()

    def forward(self, src, tgt, src_mask, tgt_mask, labels):
        # B, 1, tok_len -> B, tok_len
        src = torch.squeeze(src, 1).to(self.device)
        src_mask = torch.squeeze(src_mask, 1).to(self.device)
        tgt = torch.squeeze(tgt, 1).to(self.device)
        tgt_mask = torch.squeeze(tgt_mask, 1).to(self.device)
        
        return self.model(input_ids=src,
                          attention_mask=src_mask,
                          decoder_input_ids=tgt,
                          decoder_attention_mask=tgt_mask, return_dict=True, labels=labels)
        
    def generate(self, *args, **kwargs):
      return self.model.generate(*args, **kwargs)