from typing import NamedTuple


class ConveRTModelConfig(NamedTuple):
    num_embed_hidden: int = 512
    feed_forward1_hidden: int = 2048
    feed_forward2_hidden: int = 1024
    num_attention_project: int = 64
    vocab_size: int = 10000
    num_encoder_layers: int = 6
    dropout_rate: float = 0.1


class ConveRTTrainConfig(NamedTuple):
    batch_size: int = 64
    split_size: int = 8
    learning_rate: float = 2e-5


class ConveRTDataConfig(NamedTuple):
    sp_model_path: str
    train_dataset_dir: str
    test_dataset_dir: str
