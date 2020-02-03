import time
import torch
from torch.nn import DataParallel

from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import ConveRTTrainConfig
from .model import ConveRTCosineLoss, ConveRTDualEncoder
from .logger import TrainLogger
from typing import Union

CPU_DEVICE = torch.device("cpu")


class ConveRTTrainer:
    def __init__(
        self,
        model: Union[ConveRTDualEncoder, DataParallel],
        criterion: Union[ConveRTCosineLoss, DataParallel],
        train_config: ConveRTTrainConfig,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        logger: TrainLogger,
        device: torch.device,
    ):
        self.train_config: ConveRTTrainConfig = train_config
        self.logger: TrainLogger = logger

        self.model: Union[ConveRTDualEncoder, DataParallel] = model
        self.device: torch.device = device

        self.criterion: Union[ConveRTCosineLoss, DataParallel] = criterion
        self.optimizer = Adam(self.model.parameters(), lr=train_config.learning_rate)

        self.train_dataloader: DataLoader = train_dataloader
        self.test_dataloader: DataLoader = test_dataloader

    def train(self):
        for epoch_id in range(self.train_config.epochs):
            self.train_epoch(epoch_id)
            self.evaluation(epoch_id)
            self.save_model(epoch_id)

    def train_epoch(self, epoch_id: int):
        total_steps = len(self.train_dataloader)
        for step_id, feature in enumerate(self.train_dataloader):
            start_time = time.time()
            feature.to(self.device)

            self.optimizer.zero_grad()
            context_embed, reply_embed = self.model.forward(feature.context, feature.reply)
            step_output = self.criterion.forward(context_embed, reply_embed)
            step_output.loss.backward()
            self.optimizer.step()

            eta = (int(time.time() - start_time)) * (total_steps - step_id - 1)
            self.logger.log_train_step(epoch_id, step_id, eta, step_output)

    def evaluation(self, epoch_id):
        self.model.eval()
        total_correct, total_size = 0, 0

        for feature in self.test_dataloader:
            feature.to(self.device)

            with torch.no_grad():
                context_embed, reply_embed = self.model.forward(feature.context, feature.reply)
                eval_step_output = self.criterion.forward(context_embed, reply_embed)

            total_correct += eval_step_output.correct_count
            total_size += eval_step_output.total_count

        accuracy = float(total_correct) / total_size
        self.logger.log_eval_step(epoch_id, accuracy)

    def save_model(self, epoch_id: int):
        self.model.to(CPU_DEVICE)
