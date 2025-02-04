import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# Copy all model classes (GPT, Block, MLP, CausalSelfAttention) from train_gpt.py
# ... (copy the model classes here) 