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
    sp_model_path: str = "data/en.wiki.bpe.vs10000.model"
    train_dataset_path: str = "data/sample-dataset_train.json"
    test_dataset_path: str = "data/sample-dataset_val.json"

    model_save_dir: str = "logs/models/"
    log_dir: str = "logs"
    device: str = "cuda:0"
    # device: str = "cpu"
    use_data_paraller: bool = True

    is_reddit: bool = True

    train_batch_size: int = 64   # 64
    test_batch_size: int = 24    # 256

    split_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 100
