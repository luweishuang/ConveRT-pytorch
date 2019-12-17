import torch
from convert.attention import SelfAttention
from convert.config import ConveRTModelConfig


def test_self_attention():
    BATCH_SIZE, SEQ_LEN = 4, 5
    config = ConveRTModelConfig()
    attention = SelfAttention(config)

    query = torch.rand(BATCH_SIZE, SEQ_LEN, config.num_embed_hidden)
    attn_mask = torch.ones(query.size()[:-1], dtype=torch.float)
    output, attn = attention.forward(query, attn_mask, return_attention=True)

    assert output.size() == (BATCH_SIZE, SEQ_LEN, config.num_embed_hidden)
    assert attn.size() == (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
