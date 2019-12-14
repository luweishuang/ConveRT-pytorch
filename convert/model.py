import torch
import torch.nn as nn
import torch.nn.functional as fnn

from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import MultiheadAttention

from typing import NamedTuple


class EncoderInput(NamedTuple):
    input_ids: torch.LongTensor
    position_ids: torch.LongTensor
    attention_mask: torch.FloatTensor
    input_lengths: torch.LongTensor


class ConverRTDualEncoder(nn.Module):
    def __init__(self, config):
        self.shared_encoder = ConveRTSharedEncoder(config)
        self.context_encoder = ConveRTEncoder(self.shared_encoder)
        self.reply_encoder = ConveRTEncoder(self.shared_encoder)

    def forward(self, context_input: EncoderInput, reply_input: EncoderInput):
        context_embed = self.context_encoder.forward(context_input)
        reply_embed = self.reply_encoder.forward(reply_input)

        cosine_similarity = torch.matmul(context_embed, reply_embed.transpose(0, 1))
        # todo: scaled cosine similarities
        return cosine_similarity

    def calculate_loss(self, cosine_similarity: torch.FloatTensor):
        label = torch.arange(cosine_similarity.size(0), device=cosine_similarity.device)
        return fnn.nll_loss(cosine_similarity, label)


class ConveRTEncoder(nn.Module):
    def __init__(self, config, shared_encoder: ConveRTSharedEncoder):
        super().__init__()
        self.shared_encoder: ConveRTSharedEncoder = shared_encoder
        self.projection = nn.Linear(config.hidden_size, config.encoder_output_dim)

    def forward(self, encoder_input: EncoderInput):
        shared_encoder_output = self.shared_encoder.forward(encoder_input)
        sqrt_reduction_output = shared_encoder_output / torch.sqrt(encoder_input.input_lengths)
        encoder_output = self.projection.forward(sqrt_reduction_output)
        return encoder_output


class ConveRTSharedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = ConveRTEmbedding(config)
        self.encoder_layers = nn.Sequential([ConveRTEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.two_head_self_attn = MultiheadAttention(config.hidden_size, 2, dropout=config.dropout_rate)

    def forward(self, encoder_input: EncoderInput):
        # calculate transformer input embedding
        embed = self.embedding.forward(encoder_input.input_ids, encoder_input.position_ids)
        # run over the transformer encoder layers
        encoder_layers_output = self.encoder_layers.forward(embed, encoder_input.attention_mask)
        # pass through 2-headed self-attention
        encoder_output = self.two_head_self_attn.forward(encoder_layers_output, attn_mask=encoder_input.attention_mask)
        return encoder_output


class ConveRTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.subword_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embed = nn.Embedding(config.max_seq_len, config.hidden_size)

    def forward(self, input_ids, position_ids):
        subword_embed = self.subword_embed.forward(input_ids)
        positional_embed = self.positional_embed.forward(position_ids)
        embedding = subword_embed + positional_embed
        return embedding


class ConveRTEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiheadAttention(config.hidden_size, 1, config.dropout_rate)
        self.norm1 = LayerNorm(config.hidden_size)
        self.fead_forward = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm2 = LayerNorm(config.hidden_size)

    def forward(self, embed_output: torch.FloatTensor, attention_mask: torch.FloatTensor):
        self_attn_output = self.self_attention.forward(
            embed_output, embed_output, embed_output, attn_mask=attention_mask
        )
        norm1_output = self.norm1.forward(self_attn_output)
        fead_forward_output = self.fead_forward.forward(norm1_output)
        norm2_output = self.norm2.forward(fead_forward_output)
        return norm2_output
