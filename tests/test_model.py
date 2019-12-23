import torch

from convert.config import ConveRTModelConfig
from convert.model import (
    ConveRTDualEncoder,
    ConveRTEmbedding,
    ConveRTEncoder,
    ConveRTEncoderInput,
    ConveRTEncoderLayer,
    ConveRTSharedEncoder,
    SelfAttention,
)

config = ConveRTModelConfig()
BATCH_SIZE, SEQ_LEN = 3, 5


def test_self_attention():
    BATCH_SIZE, SEQ_LEN = 4, 5
    config = ConveRTModelConfig()
    attention = SelfAttention(config)

    query = torch.rand(BATCH_SIZE, SEQ_LEN, config.num_embed_hidden)
    attn_mask = torch.ones(query.size()[:-1], dtype=torch.float)
    output, attn = attention.forward(query, attn_mask, return_attention=True)

    assert output.size() == (BATCH_SIZE, SEQ_LEN, config.num_embed_hidden)
    assert attn.size() == (BATCH_SIZE, SEQ_LEN, SEQ_LEN)


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
    encoder_input = ConveRTEncoderInput(
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
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    encoder_output = encoder.forward(encoder_input)

    assert encoder_output.size() == (BATCH_SIZE, config.feed_forward2_hidden)


def test_convert_dual_encoder():
    dual_encoder = ConveRTDualEncoder(config)

    input_ids = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(BATCH_SIZE)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(BATCH_SIZE)], dtype=torch.float)
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    encoder_output = dual_encoder(encoder_input, encoder_input)

    assert encoder_output.size() == (BATCH_SIZE, BATCH_SIZE)


def test_convert_dual_encoder_loss():
    dual_encoder = ConveRTDualEncoder(config)

    input_ids = torch.randint(high=config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(BATCH_SIZE)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(BATCH_SIZE)], dtype=torch.float)
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    encoder_output = dual_encoder(encoder_input, encoder_input)
    cosine_similarity_loss = dual_encoder.calculate_loss(encoder_output)

    assert isinstance(cosine_similarity_loss[0].item(), float)
    assert isinstance(cosine_similarity_loss[1].item(), float)

    assert cosine_similarity_loss[0] > 0


def test_convert_dual_encoder_splited_loss():
    batch_size, split_size = 20, 5
    dual_encoder = ConveRTDualEncoder(config)

    input_ids = torch.randint(high=config.vocab_size, size=(batch_size, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(batch_size)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(batch_size)], dtype=torch.float)
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    similarity = dual_encoder.forward(encoder_input, encoder_input, split_size=split_size)
    assert similarity.size() == (batch_size // split_size, split_size, split_size)

    cosine_similarity_loss = dual_encoder.calculate_loss(similarity)

    assert isinstance(cosine_similarity_loss[0].item(), float)
    assert isinstance(cosine_similarity_loss[1].item(), float)

    assert cosine_similarity_loss[0] > 0


def test_context_reply_multiplication():
    embedding = torch.rand(BATCH_SIZE, config.feed_forward2_hidden, dtype=torch.float)
    label = torch.arange(BATCH_SIZE, dtype=torch.long)

    consine_sim = torch.matmul(embedding, embedding.transpose(-1, -2))
    output_argmax = consine_sim.argmax(-1)

    assert output_argmax.size()[0] == BATCH_SIZE
    assert output_argmax.eq(label).all().item() is True


def test_context_reply_split_multiplication():
    split_size = 8
    batch_size = BATCH_SIZE * split_size

    embedding = torch.rand(batch_size, config.feed_forward2_hidden, dtype=torch.float)
    label = torch.arange(split_size, dtype=torch.long).repeat(batch_size // split_size)

    split_embedding = embedding.view(batch_size // split_size, split_size, -1)
    split_cosine_sim = torch.matmul(split_embedding, split_embedding.transpose(-1, -2))

    reduced_cosine_sim = split_cosine_sim.view(batch_size, split_size)

    output_argmax = reduced_cosine_sim.argmax(-1)

    assert output_argmax.size()[0] == batch_size
    assert output_argmax.eq(label).all().item() is True
