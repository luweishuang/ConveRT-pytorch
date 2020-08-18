import time
from typing import Union

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import ConveRTTrainConfig
from .criterion import ConveRTCosineLoss, calculate_query_reply_matching_loss
from .logger import TrainLogger
from .model import ConveRTDualEncoder

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
        self.device_count: int = torch.cuda.device_count() if self.train_config.use_data_paraller else 1
        if not torch.cuda.is_available():                     # cpu
            self.device_count = 1

    def train(self):
        for epoch_id in range(self.train_config.epochs):
            print("epoch_id = ", epoch_id)
            self.train_epoch(epoch_id)
            self.evaluation(epoch_id)
            self.save_model(epoch_id)

    def train_epoch(self, epoch_id: int):
        total_steps = len(self.train_dataloader)
        for step_id, feature in enumerate(self.train_dataloader):
            print("step_id = ", step_id)
            start_time = time.time()

            self.optimizer.zero_grad()
            context_embed, reply_embed = self.model.forward(feature.context, feature.reply)
            query_reply_similarity = self.criterion.forward(context_embed, reply_embed)
            print("criterion.forward")
            loss, correct_count, total_count = calculate_query_reply_matching_loss(
                query_reply_similarity, self.train_config.split_size, self.device_count
            )
            print("calculate_query_reply_matching_loss")
            accuracy = float(correct_count) / total_count

            loss.backward()
            self.optimizer.step()

            eta = (int(time.time() - start_time)) * (total_steps - step_id - 1)
            self.logger.log_train_step(epoch_id, step_id, eta, loss, accuracy)

    def evaluation(self, epoch_id):
        self.model.eval()
        total_correct, total_size = 0, 0
        total_loss = 0.0

        for feature in self.test_dataloader:
            feature.to(self.device)

            with torch.no_grad():
                context_embed, reply_embed = self.model.forward(feature.context, feature.reply)
                query_reply_similarity = self.criterion.forward(context_embed, reply_embed)
                loss, correct_count, total_count = calculate_query_reply_matching_loss(
                    query_reply_similarity, self.train_config.split_size, self.device_count
                )
                accuracy = float(correct_count) / total_count
                total_loss += loss.item()

            total_correct += correct_count
            total_size += total_count

        accuracy = float(total_correct) / total_size
        avg_loss = total_loss / total_size
        self.logger.log_eval_step(epoch_id, avg_loss, accuracy)

    def save_model(self, epoch_id: int):
        self.model.to(CPU_DEVICE)
