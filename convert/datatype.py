from typing import Dict, List, NamedTuple, Union

import torch


class ConveRTEncoderInput(NamedTuple):
    input_ids: Union[torch.LongTensor, List[int]]
    attention_mask: Union[torch.FloatTensor, List[float]]
    position_ids: Union[torch.LongTensor, List[int]] = None
    input_lengths: Union[torch.LongTensor, List[int]] = None


class ConveRTFeature(NamedTuple):
    context: ConveRTEncoderInput
    reply: ConveRTEncoderInput


class ConveRTExample(NamedTuple):
    context: List[str]
    response: str

    @staticmethod
    def load_reddit_json(example: Dict[str, str]) -> "ConveRTExample":
        context_keys = sorted([key for key in example.keys() if "context" in key])
        return ConveRTExample(context=[example[key] for key in context_keys], response=example["response"],)

    @staticmethod
    def load_tsv_json(example: str) -> "ConveRTExample":
        splited_lines = example.strip().split("\t")
        return ConveRTExample(context=splited_lines[:-1], response=splited_lines[-1])


class ConveRTDualEncoderOutput(NamedTuple):
    context_embed: torch.FloatTensor
    reply_embed: torch.FloatTensor
    loss: torch.FloatTensor
    accuracy: float
    correct_count: torch.LongTensor
    total_count: int
