from typing import Dict, List, NamedTuple, Optional, Union

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
    response_author: Optional[str]
    context_author: Optional[str]
    subreddit: Optional[str]
    thread_id: Optional[str]

    @staticmethod
    def load(example: Dict[str, str]) -> "ConveRTExample":
        context_keys = sorted([key for key in example.keys() if "context" in key])
        return ConveRTExample(
            context=[example[key] for key in context_keys],
            response=example["response"],
            context_author=example.get("context_author"),
            response_author=example.get("response_author"),
            subreddit=example.get("subreddit"),
            thread_id=example.get("thread_id"),
        )
