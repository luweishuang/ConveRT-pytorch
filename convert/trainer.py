import logging
import sys
import time
from typing import Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import ConveRTTrainConfig
from .datatype import ConveRTFeature
from .model import ConveRTCosineLoss, ConveRTDualEncoder, ConveRTTrainStepOutput


class ConveRTTrainer:
    def __init__(
        self,
        model: ConveRTDualEncoder,
        train_config: ConveRTTrainConfig,
        train_data_loader: DataLoader,
        test_data_loader: Optional[DataLoader] = None,
    ):
        self.train_config: ConveRTTrainConfig = train_config
        self.logger = ConveRTTrainLogger()

        self.model: ConveRTDualEncoder = model
        self.criterion = ConveRTCosineLoss(train_config.split_size)

        self.train_data_loader: DataLoader = train_data_loader
        self.test_data_loader: DataLoader = test_data_loader

        self.optimizer = Adam(self.model.parameters(), lr=train_config.learning_rate)

    def train(self, epochs: int):
        for epoch_id in range(epochs):
            self.train_epoch(epoch_id)

            if self.test_data_loader is not None:
                accuracy = self.evaluation(epoch_id)
                self.logger.log_evaluation_output(accuracy)

            self.save_checkpoint(epoch_id)

    def train_epoch(self, epoch_id: int):
        for step_id, feature in enumerate(self.train_data_loader):
            start_time = time.time()
            step_output = self.train_step(epoch_id, step_id, feature)

            eta = (int(time.time() - start_time)) * (len(self.train_data_loader) - step_id - 1)
            self.logger.log_train_step_output(epoch_id, step_id, eta, step_output)

    def train_step(self, epoch_id: int, step_id: int, feature: ConveRTFeature) -> ConveRTTrainStepOutput:
        self.optimizer.zero_grad()
        context_embed, reply_embed = self.model.forward(feature.context, feature.reply)
        encoder_output = self.criterion.forward(context_embed, reply_embed)
        self.update_optimizer(encoder_output.loss)
        return encoder_output

    def update_optimizer(self, loss: torch.Tensor):
        loss.backward()
        self.optimizer.step()

    def evaluation(self, epoch_id) -> float:
        self.model.eval()
        total_correct, total_size = 0, 0

        for feature in self.test_data_loader:
            with torch.no_grad():
                context_embed, reply_embed = self.model.forward(feature.context, feature.reply)
                eval_step_output = self.criterion.forward(context_embed, reply_embed)

            total_correct += eval_step_output.correct_count
            total_size += eval_step_output.total_count

        return float(total_correct) / total_size

    def save_checkpoint(self, epoch_id: int, step_id: int):
        pass


class ConveRTTrainLogger:
    def __init__(self):
        super().__init__()
        self.logger = self._logger_setup()

    def log_train_step_output(self, epoch_id: int, step_id: int, eta: float, step_output: ConveRTTrainStepOutput):
        self.logger.debug(
            f"EP:{epoch_id}\tSTEP:{step_id}\t"
            f"loss:{step_output.loss}\tacc:{step_output.accuracy}\t"
            f"eta:{eta//60} min {eta%60} sec"
        )

    def log_evaluation_output(self, epoch_id: int, accuracy: float):
        self.logger.debug(f"[EVAL] EP:{epoch_id}\t eval acc: {accuracy:.4f}")

    def _logger_setup(self):
        logger = logging.Logger("convert-trainer", level=logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s\t%(message)s")

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)

        # datetime object containing current date and time
        # now = datetime.now()
        # dt_string = now.strftime("-%Y%d%m-%H%M%S.log")

        # file_handler = logging.FileHandler(log_path + dt_string)
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)

        logger.addHandler(handler)
        # logger.addHandler(file_handler)
        return logger
