import json
from typing import List

import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset

from .config import ConveRTDataConfig
from .datatype import ConveRTEncoderInput, ConveRTExample, ConveRTFeature


class ConveRTTextUtility:
    def __init__(self, config: ConveRTDataConfig):
        self.tokenizer = self._load_tokenizer(config)

    def encode_example_to_feature(self, example: ConveRTExample):
        context_str = "+".join(example.context)
        context_input = self.encode_string_to_input(context_str)
        reply_input = self.encode_string_to_input(example.response)
        return ConveRTFeature(context=context_input, reply=reply_input)

    def encode_string_to_input(self, input_str: str) -> ConveRTEncoderInput:
        input_ids = self.encode_string_to_tokens(input_str)
        attention_mask = [1 for _ in range(len(input_ids))]
        position_ids = [i for i in range(len(input_ids))]

        return ConveRTEncoderInput(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, input_lengths=len(input_ids)
        )

    def encode_string_to_tokens(self, source_str: str) -> List[int]:
        return self.tokenizer.EncodeAsIds(source_str)

    @staticmethod
    def _load_tokenizer(config: ConveRTDataConfig) -> SentencePieceProcessor:
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(config.sp_model_path)
        return tokenizer


class ConveRTDataset(Dataset):
    def __init__(self, examples: List[ConveRTExample], text_utiltity: ConveRTTextUtility):
        super().__init__()
        self.examples = examples
        self.text_utility = text_utiltity

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item) -> ConveRTFeature:
        return self.text_utility.encode_example_to_feature(self.examples[item])

    @staticmethod
    def concat_tensor_by_attribute(
        encoder_inputs: List[ConveRTEncoderInput], attr_name: str, add_padding: bool = True
    ) -> torch.Tensor:
        concat_candidate = [getattr(encoder_input, attr_name) for encoder_input in encoder_inputs]

        if add_padding:
            max_seq_len = max(map(len, concat_candidate))
            for candidate in concat_candidate:
                candidate.extend([0 for _ in range(max_seq_len - len(candidate))])

        return torch.tensor(concat_candidate)

    @staticmethod
    def concat_encoder_inputs(encoder_inputs: List[ConveRTEncoderInput]) -> ConveRTEncoderInput:
        return ConveRTEncoderInput(
            input_ids=ConveRTDataset.concat_tensor_by_attribute(encoder_inputs, "input_ids"),
            attention_mask=ConveRTDataset.concat_tensor_by_attribute(encoder_inputs, "attention_mask"),
            position_ids=ConveRTDataset.concat_tensor_by_attribute(encoder_inputs, "position_ids"),
            input_lengths=ConveRTDataset.concat_tensor_by_attribute(encoder_inputs, "input_lengths", add_padding=False),
        )

    @staticmethod
    def collate_fn(features: List[ConveRTFeature]) -> ConveRTFeature:
        return ConveRTFeature(
            context=ConveRTDataset.concat_encoder_inputs([feature.context for feature in features]),
            reply=ConveRTDataset.concat_encoder_inputs([feature.reply for feature in features]),
        )

    @staticmethod
    def from_reddit_dataset(dataset_path: str, text_util: ConveRTTextUtility) -> "ConveRTDataset":
        with open(dataset_path) as f:
            examples = [ConveRTExample.load_reddit_json(json.loads(line.strip())) for line in f]
        return ConveRTDataset(examples, text_util)

    @staticmethod
    def from_tsv_dataset(dataset_path: str, text_util: ConveRTTextUtility) -> "ConveRTDataset":
        with open(dataset_path) as f:
            examples = [ConveRTExample.load_tsv_line(line.strip()) for line in f]
        return ConveRTDataset(examples, text_util)
