import torch
from torch.utils.data import Dataset
import numpy as np


def make_decoder_input(x, eos_token=1, pad_token=3):
  # Use torch.roll to shift the elements to the right by one place
  x = torch.roll(x, shifts=1, dims=1)

  # Set the first column to </s>
  x[:, 0] = eos_token

  # Check where the value is '3'
  matches = (x == pad_token)

  # Use torch.where to get the indices
  indices = [torch.where(m)[0][0] if m.sum() > 0 else 0 for m in matches]

  for i, index in enumerate(indices):
    x[i, index - 1] = 3

  return x


class CustomDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer

        input_tokens = self.tokenizer(self.inputs, padding="max_length", max_length=30, truncation=True, return_tensors='pt')
        output_tokens = self.tokenizer(self.outputs, padding="max_length", max_length=30, truncation=True, return_tensors='pt')

        self.input_ids = input_tokens['input_ids']
        self.input_mask = input_tokens['attention_mask']
        
        self.labels = output_tokens['input_ids']
        self.output_mask = output_tokens['attention_mask']

        self.output_ids = make_decoder_input(self.labels)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx], self.input_mask[idx], self.output_mask[idx], self.labels[idx]
