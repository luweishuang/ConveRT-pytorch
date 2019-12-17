import torch

from convert.config import ConveRTModelConfig
from convert.model import (
    ConveRTDualEncoder,
    ConveRTEmbedding,
    ConveRTEncoder,
    ConveRTEncoderLayer,
    ConveRTSharedEncoder,
    EncoderInput,
)

config = ConveRTModelConfig()
BATCH_SIZE, SEQ_LEN = 3, 5


def test_convert_embedding():
    embedding = ConveRTEmbedding(config)
    model_input = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    embedding_output = embedding.forward(input_ids=model_input)

    assert embedding_output.size() == (BATCH_SIZE, SEQ_LEN, config.num_embed_hidden)


def test_convert_encoder_layer():
    embedding = ConveRTEmbedding(config)
    encoder_layer = ConveRTEncoderLayer(config)

    model_input = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    embedding_output = embedding.forward(input_ids=model_input)
    encoder_layer_output = encoder_layer.forward(embedding_output)

    assert encoder_layer_output.size() == (BATCH_SIZE, SEQ_LEN, config.num_embed_hidden)


def test_convert_shared_encoder():
    shared_encoder = ConveRTSharedEncoder(config)

    input_ids = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(BATCH_SIZE)])
    encoder_input = EncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=None
    )

    encoder_output = shared_encoder(encoder_input)

    assert encoder_output.size() == (BATCH_SIZE, SEQ_LEN, config.num_embed_hidden)


def test_convert_encoder():
    shared_encoder = ConveRTSharedEncoder(config)
    encoder = ConveRTEncoder(config, shared_encoder)

    input_ids = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(BATCH_SIZE)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(BATCH_SIZE)], dtype=torch.float)
    encoder_input = EncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    encoder_output = encoder.forward(encoder_input)

    assert encoder_output.size() == (BATCH_SIZE, config.feed_forward2_hidden)


def test_convert_dual_encoder():
    dual_encoder = ConveRTDualEncoder(config)

    input_ids = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(BATCH_SIZE)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(BATCH_SIZE)], dtype=torch.float)
    encoder_input = EncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    encoder_output = dual_encoder(encoder_input, encoder_input)

    assert encoder_output.size() == (BATCH_SIZE, BATCH_SIZE)


def test_convert_dual_encoder_loss():
    dual_encoder = ConveRTDualEncoder(config)

    input_ids = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(BATCH_SIZE)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(BATCH_SIZE)], dtype=torch.float)
    encoder_input = EncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    encoder_output = dual_encoder(encoder_input, encoder_input)
    cosine_similarity_loss = dual_encoder.calculate_loss(encoder_output)

    assert isinstance(cosine_similarity_loss["loss"].item(), float)
    assert isinstance(cosine_similarity_loss["acc"].item(), float)

    assert cosine_similarity_loss["loss"] > 0
