import logging
from torch.utils.tensorboard import SummaryWriter
from .model import ConveRTTrainStepOutput


class TrainLogger(logging.Logger):
    def __init__(self, logger_name: str, tensorboard: SummaryWriter):
        self.tensorboard: SummaryWriter = tensorboard
        self.global_train_step = 0
        self.global_eval_step = 0
        super().__init__(logger_name)

    def log_train_step(self, epoch_id: int, step_id: int, eta: float, step_output: ConveRTTrainStepOutput):
        self.info(
            f"EP:{epoch_id}\tSTEP:{step_id}\t"
            f"loss:{step_output.loss}\tacc:{step_output.accuracy}\t"
            f"eta:{eta//60} min {eta%60} sec"
        )
        self.tensorboard.add_scalar("train/acc", step_output.accuracy)

    def log_eval_step(self, epoch_id: int, accuracy: float):
        self.info(f"[EVAL] EP:{epoch_id}\t eval acc: {accuracy:.4f}")
        self.tensorboard.add_scalar("eval/acc", accuracy)
