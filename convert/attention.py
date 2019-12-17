import torch.nn as nn
import torch
from .config import ConveRTModelConfig
import math


class SelfAttention(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()

        self.config = config
        self.query = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.key = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.value = nn.Linear(config.num_embed_hidden, config.num_attention_project)

        self.softmax = nn.Softmax(dim=-1)
        self.output_projection = nn.Linear(config.num_attention_project, config.num_embed_hidden)

    def forward(self, query, attention_mask=None, return_attention=False):
        _query = self.query.forward(query)
        _key = self.key.forward(query)
        _value = self.value.forward(query)

        # scaled dot product (https://www.aclweb.org/anthology/N18-2074.pdf Fig.2)
        attention_scores = torch.matmul(_query, _key.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(self.config.num_attention_project)

        if attention_mask is not None:
            extended_attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask.unsqueeze(-1)) * -10000.0
            attention_scores = attention_scores + extended_attention_mask

        attention_weights = self.softmax(attention_scores)
        weighted_value = torch.matmul(attention_weights, _value)

        output_value = self.output_projection.forward(weighted_value)

        return (output_value, attention_weights) if return_attention else output_value
