from typing import List
import torch
import torch.nn as nn
import numpy as np


MODEL_CONFIG = {
    "n_users": 610,
    "n_items": 9724,
    "dim_size": 32,
    "sigmoid": True,
    "bias": True,
    "init": True,
}


def round_to_0p5(list_nums: List[float]):
    """ Helper func to round nums to nearest 0.5, eg 1.45 -> 1.5 """
    return np.round(np.array(list_nums)*2)/2


def sigmoid_range(x, low, high):
    return torch.sigmoid(x) * (high - low) + low


class MFAdvanced(nn.Module):
    """ Matrix factorization + user & item bias, weight init., sigmoid_range """
    def __init__(self, num_users: int, num_items: int, emb_dim: int, init: bool, bias: bool, sigmoid: bool):
        super().__init__()
        self.bias = bias
        self.sigmoid = sigmoid
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        if bias:
            self.user_bias = nn.Parameter(torch.zeros(num_users))
            self.item_bias = nn.Parameter(torch.zeros(num_items))
            self.offset = nn.Parameter(torch.zeros(1))
        if init:
            self.user_emb.weight.data.uniform_(0., 0.05)
            self.item_emb.weight.data.uniform_(0., 0.05)

    def forward(self, user: torch.tensor, item: torch.tensor):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)

        element_product = (user_emb*item_emb).sum(1)

        if self.bias:
            user_b = self.user_bias[user]
            item_b = self.item_bias[item]
            element_product += user_b + item_b + self.offset
        if self.sigmoid:
            return sigmoid_range(element_product, 0, 5.5)
        return element_product
