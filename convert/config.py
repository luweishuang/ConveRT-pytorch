from typing import NamedTuple


class ConveRTModelConfig(NamedTuple):
    num_embed_hidden: int = 512
    feed_forward1_hidden: int = 2048
    feed_forward2_hidden: int = 1024
    num_attention_project: int = 64
    vocab_size: int = 64
    num_encoder_layers: int = 6
    dropout_rate: float = 0.1
