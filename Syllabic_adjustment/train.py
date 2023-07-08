import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import BartForConditionalGeneration
from kobart import get_kobart_tokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd

import logging

from dataset import CustomDataset
from Syllabic_adjustment_model import CustomKoBART

import argparse
from datetime import datetime



def dataset_split(dataset, ratio):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


########################################################################
# reference: https://github.com/Seoneun/KoBART-Question-Generation
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# dev
def _validate(
        model: BartForConditionalGeneration,
        dev_dataloader: DataLoader,
        logger: logging.Logger,
        global_step: int,
):
    model.eval()
    loss_list = []
    for batch_data in tqdm(dev_dataloader, desc="[EVAL]"):
        with torch.no_grad():
            src, tgt, src_mask, tgt_mask, labels = batch_data
            model_outputs = model(src, tgt, src_mask, tgt_mask, labels)
            loss_list.append(model_outputs.loss.item())

    mean_loss = np.mean(loss_list)
    logger.info(f"[EVAL] global_step:{global_step} loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}")
    model.train()

    return mean_loss

def train(model, epochs, optimizer, scheduler, train_loader, val_loader, grad_clip):
  checkpoint_dir = ''
  model.train()
  loss_list_between_log_interval = []
  for epoch_id in range(epochs):
      for step_index, batch_data in tqdm(enumerate(train_loader), f"[TRAIN] EP:{epoch_id}", total=len(train_loader)):
              global_step = len(train_loader) * epoch_id + step_index + 1
              optimizer.zero_grad()

              src, tgt, src_mask, tgt_mask, labels = batch_data

              model_outputs = model(src, tgt, src_mask, tgt_mask, labels)
              loss = model_outputs['loss']
              loss.backward()

              # model_outputs.loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
              optimizer.step()
              scheduler.step()

              # for logging
              loss_list_between_log_interval.append(loss.item())



      mean_loss = np.mean(loss_list_between_log_interval)
      logger.info(
          f"EP:{epoch_id} global_step:{global_step} "
          f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}"
      )
      loss_list_between_log_interval.clear()

      dev_loss = _validate(model, val_loader, logger, global_step)

      if epoch_id < 1:
        now = datetime.now().strftime('%x %X')  # MM/DD/YY HH:mm:ss
        checkpoint_dir = f'../results/{now}_checkpoints'
        os.makirs(checkpoint_dir, exist_ok=True)

      state_dict = model.state_dict()
      model_path = os.path.join(checkpoint_dir, f"syllabic_adjustment_{global_step}.pth")
      logger.info(f"global_step: {global_step} model saved at {model_path}")
      torch.save(state_dict, model_path)
############################################################################



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train Syllabic Adjustment model')

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, defulat=3e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--train_csv_file', type=str)
    parser.add_argument('--checkpoint_path', type=str, default=None)

    args = parser.parse_args()

    DATASET_PATH = os.path.join('../dataset', args.train_csv_file)


    # data
    df = pd.read_csv(DATASET_PATH)

    lines1 = list(df['X_w_tokens']) # X
    lines2 = list(df['GT']) # GT

    kobart_tokenizer = get_kobart_tokenizer()
    dataset = CustomDataset(lines1, lines2, kobart_tokenizer)
    train_dataset, val_dataset = dataset_split(dataset, 0.8)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # model
    model = CustomKoBART()
    if args.checkpoint_path:     # load checkpoint
        model.load_state_dict(torch.load(args.checkpoint_path))

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    # optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # scheduler
    data_len = len(train_loader)
    num_train_steps = int(data_len / args.batch_size * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    # logging data info
    logging.info(f'data length {data_len}')
    logging.info(f'num_train_steps : {num_train_steps}')
    logging.info(f'num_warmup_steps : {num_warmup_steps}')

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(model, args.epochs, optimizer, scheduler, train_loader, val_loader, args.grad_clip)