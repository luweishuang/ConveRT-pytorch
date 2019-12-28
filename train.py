from torch.utils.data import DataLoader

from convert.config import ConveRTDataConfig, ConveRTModelConfig, ConveRTTrainConfig
from convert.dataset import ConveRTDataset, ConveRTTextUtility
from convert.model import ConveRTDualEncoder
from convert.trainer import ConveRTTrainer


def train():
    dataset_path = "data/sample-dataset.json"
    sp_model_path = "data/en.wiki.bpe.vs10000.model"

    data_config = ConveRTDataConfig(sp_model_path=sp_model_path, train_dataset_dir=None, test_dataset_dir=None)
    train_config = ConveRTTrainConfig()
    model_config = ConveRTModelConfig()

    text_utility = ConveRTTextUtility(data_config)
    dataset = ConveRTDataset.from_reddit_dataset(dataset_path, text_utility)
    train_data_loader = DataLoader(dataset, batch_size=train_config.batch_size, collate_fn=dataset.collate_fn)

    model = ConveRTDualEncoder(model_config)
    trainer = ConveRTTrainer(model, train_config, train_data_loader=train_data_loader)
    trainer.train(10)


if __name__ == "__main__":
    train()
