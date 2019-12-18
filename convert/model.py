import math
from typing import NamedTuple, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import MultiheadAttention

from .config import ConveRTModelConfig


class EncoderInput(NamedTuple):
    input_ids: torch.LongTensor
    attention_mask: torch.FloatTensor
    position_ids: torch.LongTensor = None
    input_lengths: torch.LongTensor = None


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
        """[summary]
        
        :param config: model config
        :type config: ConveRTModelConfig
        """
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
    """ Shared(context, reply) encoder for generate sentence representation.
        This shared encoder will be used in ConveRTEncoder.__init__ input argument. 
        
        SharedEncoder is consisted with under layers.

        1. sub-word + positional embedding layer
        2. multi-layer (6-layers on paper) ConveRTEncoderLayer (transformer block)
        3. 2-head self-attention layer

        It doesn't forward feed-forward-2 as described in paper figure 2. 
    """

    def __init__(self, config: ConveRTModelConfig):
        """ initialize model with config value

        :param config: model config
        :type config: ConveRTModelConfig
        """
        super().__init__()
        self.embedding = ConveRTEmbedding(config)
        self.encoder_layers = nn.ModuleList([ConveRTEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.two_head_self_attn = MultiheadAttention(config.num_embed_hidden, 2, dropout=config.dropout_rate)

    def forward(self, encoder_input: EncoderInput) -> torch.Tensor:
        """ Make sentence representation with under procedure

        1. pass to sub-word embedding (subword and positional)
        2. pass to mulit-layer transformer block
        3. pass to 2-head self-attention layer

        :param encoder_input: raw encoder inputs (EncoderInput.input_ids, EncoderInput.position_ids etc..)
        :type encoder_input: EncoderInput
        :return: sentence representation (which didn't pass through fead-forward-2 layer)
        :rtype: torch.Tensor
        """

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
    """ Seperated(context, reply) encoder for making sentence representation. Encoder is consisted with under layers.

    1. shared_encoder (shared with context, reply encoder)
    2. fead_forward (feed-forward-2 layer on figure 2 3-layer fully connected feed forward net)
    """

    def __init__(self, config: ConveRTModelConfig, shared_encoder: ConveRTSharedEncoder):
        """ Initialize model with config value
        Additionally, shared_encoder which is initialized seperately, should be pass through the argument.

        :param config: model config
        :type config: ConveRTModelConfig
        :param shared_encoder: shared encoder model for making sentence representation before fead-forward-2
        :type shared_encoder: ConveRTSharedEncoder
        """
        super().__init__()

        self.shared_encoder: ConveRTSharedEncoder = shared_encoder
        # todo: check linear dimension size
        self.feed_forward = ConveRTOuterFeedForward(
            config.num_embed_hidden, config.feed_forward2_hidden, config.dropout_rate
        )

    def forward(self, encoder_input: EncoderInput) -> torch.Tensor:
        """ Make each sentence representation (context, reply) by following procedure.

        1. pass through shared encoder -> 1-step sentence represnetation
        2. summation each sequence output into single sentence representation (SEQ_LEN, HIDDEN) -> (HIDDEN)
        3. normalize sentence representation with each sequence length (reduce reduction problem of diffrent sentence length)
        4. pass throught the feed-foward-2 layer which has independent weight by context and reply encoder part.

        :param encoder_input: raw model input (EncoderInput.input_ids, EncoderInput.position_ids, etc)
        :type encoder_input: EncoderInput
        :return: sentence representation about the context or reply
        :rtype: torch.Tensor
        """
        shared_encoder_output = self.shared_encoder.forward(encoder_input)
        sumed_word_representations = shared_encoder_output.sum(1)

        input_lengths = encoder_input.input_lengths.view(-1, 1)
        sqrt_reduction_output = sumed_word_representations / torch.sqrt(input_lengths)

        encoder_output = self.feed_forward.forward(sqrt_reduction_output)
        return encoder_output


class ConveRTDualEncoder(nn.Module):
    """ DualEncoder calculate similairty between context and reply by dot-product.

    DualEncoder is consisted with under models
    
    1. shared_encoder (shared with context, reply encoder)
    2. context_encoder (sentence representation encoder for context input)
    3. reply_encoder (sentence representation encoder for reply input)
    """

    def __init__(self, config: ConveRTModelConfig):
        """ Initialize model with config value
        shared_encoder is created in __init__ with ConveRTModelConfig and distributed to each encoder. 

        :param config: [description]
        :type config: ConveRTModelConfig
        """
        super().__init__()
        self.shared_encoder = ConveRTSharedEncoder(config)
        self.context_encoder = ConveRTEncoder(config, self.shared_encoder)
        self.reply_encoder = ConveRTEncoder(config, self.shared_encoder)

    def forward(
        self,
        context_input: EncoderInput,
        reply_input: EncoderInput,
        use_softmax: bool = False,
        with_embed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ calculate similarity matrix (CONTEXT_BATCH_SIZE, REPLY_BATCH_SIZE) between context and reply
        
        :param context_input: raw context encoder input
        :type context_input: EncoderInput
        :param reply_input: raw reply encoder input
        :type reply_input: EncoderInput
        :param use_softmax: apply softmax on similarity matrix or not, defaults to False
        :type use_softmax: bool, optional
        :param with_embed: return embedding value or not, defaults to False
        :type with_embed: bool, optional
        :return: (CONTEXT_BATCH_SIZE, REPLY_BATCH_SIZE) size of similarity_matrix + (context_embed, reply_embed)
        :rtype: -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        context_embed = self.context_encoder.forward(context_input)
        reply_embed = self.reply_encoder.forward(reply_input)
        cosine_sim = self.calculate_similarity(context_embed, reply_embed, use_softmax)
        return (cosine_sim, context_embed, reply_embed) if with_embed else cosine_sim

    def calculate_similarity(
        self, context_embed: torch.Tensor, reply_embed: torch.Tensor, use_softmax: bool = False
    ) -> torch.Tensor:
        """ calculate similairty between two matrix using dot-product
        
        :param context_embed: context representation (BATCH, HIDDEN_DIM)
        :type context_embed: torch.Tensor
        :param reply_embed: reply representation (BATCH, HIDDEN_DIM)
        :type reply_embed: torch.Tensor
        :param use_softmax: apply softmax on similarity matrix or not, defaults to False
        :type use_softmax: bool, optional
        :return: dot-product output of two matrix
        :rtype: torch.Tensor
        """
        # TODO : Scaled-Dot Product
        cosine_similarity = torch.matmul(context_embed, reply_embed.transpose(-1, -2))
        return fnn.softmax(cosine_similarity, dim=-1) if use_softmax else cosine_similarity

    def calculate_loss(self, cosine_similarity: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ calculate context-reply matching loss with categorical-cross entropy.

        :param cosine_similarity: cosine similairty matrix (CONTEXT_BATCH_SIZE, REPLY_BATCH_SIZE)
        :type cosine_similarity: torch.FloatTensor
        :return: backward available loss and accuracy
        :rtype: torch.Tensor, torch.Tensor
        """
        label = torch.arange(cosine_similarity.size(0), device=cosine_similarity.device)
        loss = fnn.cross_entropy(cosine_similarity, label)
        accuracy = cosine_similarity.argmax(-1).eq(label).float().sum() / label.size(0)
        return loss, accuracy
