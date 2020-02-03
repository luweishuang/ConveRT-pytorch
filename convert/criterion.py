from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fnn


class ConveRTTrainStepOutput(NamedTuple):
    context_embed: torch.Tensor
    reply_embed: torch.Tensor
    loss: torch.Tensor
    accuracy: float
    correct_count: int
    total_count: int


class ConveRTCosineLoss(nn.Module):
    def __init__(self, split_size: int = 1):
        """calculate similarity matrix (CONTEXT_BATCH_SIZE, REPLY_BATCH_SIZE) between context and reply

        :param split_size: split matrix into fixed-size, defaults to 1
        :type split_size: int, optional
        """
        super().__init__()
        self.split_size = split_size

    def forward(self, context_embed: torch.Tensor, reply_embed: torch.Tensor) -> ConveRTTrainStepOutput:
        """calculate context-reply matching loss using negative-sample in batch.
        if query - reply combination is 100, then each negative sample of query is 99.
        so we multiply query and reply embedding into 100 x 100 similiarity matrix.
        and then calculate the loss with 1-100 continueous label.

        :param context_embed: encoded context embedding
        :type context_embed: torch.Tensor
        :param reply_embed: encoded reply embedding
        :type reply_embed: torch.Tensor
        :return: computed loss, acc, etc
        :rtype: ConveRTTrainStepOutput
        """
        cosine_similarity = calculate_query_reply_similarity(context_embed, reply_embed, split_size=self.split_size)
        loss, correct_count, total_count = calculate_query_reply_matching_loss(cosine_similarity)
        accuracy = float(correct_count) / total_count

        return ConveRTTrainStepOutput(
            context_embed=context_embed,
            reply_embed=reply_embed,
            loss=loss,
            accuracy=accuracy,
            correct_count=correct_count,
            total_count=total_count,
        )


def calculate_query_reply_similarity(
    context_embed: torch.Tensor, reply_embed: torch.Tensor, split_size: int = 1
) -> torch.Tensor:
    """ calculate similairty between two matrix using dot-product

        :param context_embed: context representation (BATCH, HIDDEN_DIM)
        :type context_embed: torch.Tensor
        :param reply_embed: reply representation (BATCH, HIDDEN_DIM)
        :type reply_embed: torch.Tensor
        :param use_softmax: apply softmax on similarity matrix or not, defaults to False
        :type use_softmax: bool, optional
        :param split_size: split context and reply into split_size to calculate cosine similarity in fixed-length.
        :type split_size: int, optional
        :return: dot-product output of two matrix (CONTEXT_BATCH_SIZE, REPLY_BATCH_SIZE)
        :rtype: torch.Tensor
        """
    # TODO : Scaled-Dot Product
    assert context_embed.size(0) == reply_embed.size(0)

    if split_size > 1:
        assert context_embed.size(0) % split_size == 0
        context_embed = context_embed.view(context_embed.size(0) // split_size, split_size, -1)
        reply_embed = reply_embed.view(reply_embed.size(0) // split_size, split_size, -1)

    cosine_similarity = torch.matmul(context_embed, reply_embed.transpose(-1, -2))
    return cosine_similarity


def calculate_query_reply_matching_loss(cosine_similarity: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """calculate context-reply matching loss with categorical-cross entropy.

        :param cosine_similarity: cosine similairty matrix (CONTEXT_BATCH_SIZE, REPLY_BATCH_SIZE)
        :type cosine_similarity: torch.Tensor
        :return: loss, correct_count, total_size
        :rtype: Tuple[torch.Tensor, int, int]
        """
    is_splited = len(cosine_similarity.size()) == 3
    label_batch_size = cosine_similarity.size(1) if is_splited else cosine_similarity.size(0)
    label = torch.arange(label_batch_size, device=cosine_similarity.device)

    if is_splited:
        splited_batch_size, split_size = cosine_similarity.size(0), cosine_similarity.size(1)
        label = label.repeat(cosine_similarity.size(0)).view(-1)
        cosine_similarity = cosine_similarity.view(splited_batch_size * split_size, split_size)

    loss = fnn.cross_entropy(cosine_similarity, label)
    correct_count = int(cosine_similarity.argmax(-1).eq(label).long().sum().item())
    return loss, correct_count, label.size(0)
