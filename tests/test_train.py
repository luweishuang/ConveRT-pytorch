import pytest
import torch
from torch.utils.data import DataLoader

from convert.config import ConveRTDataConfig, ConveRTModelConfig, ConveRTTrainConfig
from convert.dataset import ConveRTDataset, ConveRTTextUtility
from convert.model import ConveRTDualEncoder
from convert.trainer import ConveRTTrainer


@pytest.fixture
def train_data_loader():
    data_config = ConveRTDataConfig(
        sp_model_path="data/en.wiki.bpe.vs10000.model", train_dataset_dir=None, test_dataset_dir=None
    )
    train_config = ConveRTTrainConfig()
    text_utility = ConveRTTextUtility(data_config)

    dataset = ConveRTDataset.from_reddit_dataset("data/sample-dataset.json", text_utility)
    data_loader = DataLoader(dataset, batch_size=train_config.batch_size, collate_fn=dataset.collate_fn)
    return data_loader


@pytest.fixture
def convert_model():
    # small model config for testing
    model_config = ConveRTModelConfig(
        num_embed_hidden=32,
        feed_forward1_hidden=64,
        feed_forward2_hidden=32,
        num_attention_project=16,
        vocab_size=10000,
        num_encoder_layers=2,
        dropout_rate=0.1,
    )
    return ConveRTDualEncoder(model_config)


def test_init_trainer(convert_model: ConveRTDualEncoder, train_data_loader: DataLoader):
    train_config = ConveRTTrainConfig()
    trainer = ConveRTTrainer(convert_model, train_config, train_data_loader)
    assert trainer is not None


def test_train_one_step(convert_model: ConveRTDualEncoder, train_data_loader: DataLoader):
    train_config = ConveRTTrainConfig(batch_size=32, split_size=8, learning_rate=2e-3)
    trainer = ConveRTTrainer(convert_model, train_config, train_data_loader)
    step_output = trainer.train_step(epoch_id=1, step_id=1, feature=next(iter(train_data_loader)))

    assert step_output.loss.item() > 0
    assert step_output.accuracy >= 0
    assert step_output.correct_count >= 0

    assert isinstance(step_output.reply_embed, torch.FloatTensor)
    assert isinstance(step_output.context_embed, torch.FloatTensor)


def test_train_one_step_logging(convert_model: ConveRTDualEncoder, train_data_loader: DataLoader):
    train_config = ConveRTTrainConfig(batch_size=32, split_size=8, learning_rate=2e-3)
    trainer: ConveRTTrainer = ConveRTTrainer(convert_model, train_config, train_data_loader)
    step_output = trainer.train_step(epoch_id=1, step_id=1, feature=next(iter(train_data_loader)))
    trainer.logger.log_train_step(epoch_id=1, step_id=2, step_output=step_output, eta=10.0)


def test_train_one_epoch(convert_model: ConveRTDualEncoder, train_data_loader: DataLoader):
    train_config = ConveRTTrainConfig(batch_size=128, split_size=8, learning_rate=2e-3)
    trainer: ConveRTTrainer = ConveRTTrainer(convert_model, train_config, train_data_loader)
    trainer.train_epoch(epoch_id=1)
