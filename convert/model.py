import torch
import torch.nn as nn
import torch.nn.functional as fnn

from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import MultiheadAttention

from typing import NamedTuple, Optional
from .config import ConveRTModelConfig
from .attention import SelfAttention


class EncoderInput(NamedTuple):
    input_ids: torch.LongTensor
    attention_mask: torch.FloatTensor
    position_ids: torch.LongTensor = None
    input_lengths: torch.LongTensor = None


class ConveRTInnerFeedForward(nn.Module):
    def __init__(self, input_hidden, intermediate_hidden, dropout_rate: float = None):
        super().__init__()
        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, input_hidden)

    def forward(self, x):
        x = fnn.relu(self.linear_1(x))
        return self.linear_2(self.dropout(x))


class ConveRTOuterFeedForward(nn.Module):
    def __init__(self, input_hidden, intermediate_hidden, dropout_rate: float = None):
        super().__init__()
        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(intermediate_hidden, intermediate_hidden)
        self.norm = LayerNorm(intermediate_hidden)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(self.dropout(x))
        x = self.linear_3(self.dropout(x))
        return fnn.gelu(self.norm(x))


class ConveRTEmbedding(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        # embedding dimensionality of 512.
        self.subword_embed = nn.Embedding(config.vocab_size, config.num_embed_hidden)
        self.m1_positional_embed = nn.Embedding(47, config.num_embed_hidden)
        self.m2_positional_embed = nn.Embedding(11, config.num_embed_hidden)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device)

        subword_embed = self.subword_embed.forward(input_ids)
        m1_positional_embed = self.m1_positional_embed.forward(position_ids)
        m2_positional_embed = self.m2_positional_embed.forward(position_ids)
        embedding = subword_embed + m1_positional_embed + m2_positional_embed
        return embedding


class ConveRTEncoderLayer(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        # TODO: relative position self attention
        self.self_attention = SelfAttention(config)
        self.norm1 = LayerNorm(config.num_embed_hidden)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.fead_forward = ConveRTInnerFeedForward(
            config.num_embed_hidden, config.feed_forward1_hidden, config.dropout_rate
        )
        self.norm2 = LayerNorm(config.num_embed_hidden)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, embed_output: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor] = None):
        self_attn_output = self.self_attention.forward(embed_output, attention_mask=attention_mask)
        self_attn_output = self.dropout1(self_attn_output)
        norm1_output = self.norm1.forward(self_attn_output + embed_output)

        feed_forward_output = self.fead_forward.forward(norm1_output)
        feed_forward_output = self.dropout2(feed_forward_output)
        norm2_output = self.norm2.forward(feed_forward_output + norm1_output)
        return norm2_output


class ConveRTSharedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = ConveRTEmbedding(config)
        self.encoder_layers = nn.ModuleList([ConveRTEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.two_head_self_attn = MultiheadAttention(config.num_embed_hidden, 2, dropout=config.dropout_rate)

    def forward(self, encoder_input: EncoderInput):
        # calculate transformer input embedding
        embed = self.embedding.forward(encoder_input.input_ids, encoder_input.position_ids)

        # run over the transformer encoder layers
        encoder_layer_output = embed
        for encoder_layer in self.encoder_layers:
            encoder_layer_output = encoder_layer.forward(encoder_layer_output, encoder_input.attention_mask)

        # pass through 2-headed self-attention
        encoder_output, _ = self.two_head_self_attn.forward(
            encoder_layer_output, encoder_layer_output, encoder_layer_output, attn_mask=encoder_input.attention_mask
        )
        return encoder_output


class ConveRTEncoder(nn.Module):
    def __init__(self, config, shared_encoder: ConveRTSharedEncoder):
        super().__init__()
        self.shared_encoder: ConveRTSharedEncoder = shared_encoder
        # todo: check linear dimension size
        self.feed_forward = ConveRTOuterFeedForward(
            config.num_embed_hidden, config.feed_forward2_hidden, config.dropout_rate
        )

    def forward(self, encoder_input: EncoderInput):
        shared_encoder_output = self.shared_encoder.forward(encoder_input)
        sumed_word_representations = shared_encoder_output.sum(1)

        input_lengths = encoder_input.input_lengths.view(-1, 1)
        sqrt_reduction_output = sumed_word_representations / torch.sqrt(input_lengths)

        encoder_output = self.feed_forward.forward(sqrt_reduction_output)
        return encoder_output


class ConveRTDualEncoder(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.shared_encoder = ConveRTSharedEncoder(config)
        self.context_encoder = ConveRTEncoder(config, self.shared_encoder)
        self.reply_encoder = ConveRTEncoder(config, self.shared_encoder)

    def forward(self, context_input: EncoderInput, reply_input: EncoderInput):
        context_embed = self.context_encoder.forward(context_input)
        reply_embed = self.reply_encoder.forward(reply_input)

        cosine_similarity = torch.matmul(context_embed, reply_embed.transpose(-1, -2))
        return cosine_similarity

    def calculate_loss(self, cosine_similarity: torch.FloatTensor):
        label = torch.arange(cosine_similarity.size(0), device=cosine_similarity.device)
        loss = fnn.cross_entropy(cosine_similarity, label)
        acc = cosine_similarity.argmax(-1).eq(label).float().sum() / label.size(0)
        return {"loss": loss, "acc": acc}
