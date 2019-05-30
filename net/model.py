import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

from .layers import FirstOrder, SecondOrder, AttentionSecondOrder, Attention


class FactorizationMachine(BaseModel):

    def __init__(self, dims, config={}, state_dict=None):
        super(FactorizationMachine, self).__init__(config)
        k = config['model']['args']['fm_k']

        self.bias = FirstOrder(dims)
        self.sec = SecondOrder(dims, k, batch_size=batch_size)
        self.global_bias = nn.Parameter(torch.tensor(0.))

    # batch => users, items, gens, targets
    def forward(self, batch):

        bias = self.bias(batch)
        sec = self.sec(batch)
        return self.global_bias + bias + sec


class AttentionalFactorizationMachine(BaseModel):

    def __init__(self, dims, config={}, state_dict=None):
        super(AttentionalFactorizationMachine, self).__init__(config)
        k = config['model']['args']['fm_k']
        t = config['model']['args']['afm_t']

        self.bias = FirstOrder(dims)
        self.sec = AttentionSecondOrder(dims, k, t=2)
        self.global_bias = nn.Parameter(torch.tensor(0.))

    # batch => users, items, gens, targets
    def forward(self, batch):

        bias = self.bias(batch)
        sec = self.sec(batch)
        return self.global_bias + bias + sec
