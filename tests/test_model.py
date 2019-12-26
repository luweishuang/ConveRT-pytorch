import torch

from convert.config import ConveRTModelConfig, ConveRTTrainConfig
from convert.model import (
    ConveRTCosineLoss,
    ConveRTDualEncoder,
    ConveRTEmbedding,
    ConveRTEncoder,
    ConveRTEncoderInput,
    ConveRTEncoderLayer,
    ConveRTSharedEncoder,
    SelfAttention,
)

model_config = ConveRTModelConfig()
train_config = ConveRTTrainConfig(batch_size=32, split_size=8, learning_rate=2e-5)
SEQ_LEN = 5


def test_self_attention():
    config = ConveRTModelConfig()
    attention = SelfAttention(config)

    query = torch.rand(train_config.batch_size, SEQ_LEN, config.num_embed_hidden)
    attn_mask = torch.ones(query.size()[:-1], dtype=torch.float)
    output, attn = attention.forward(query, attn_mask, return_attention=True)

    assert output.size() == (train_config.batch_size, SEQ_LEN, config.num_embed_hidden)
    assert attn.size() == (train_config.batch_size, SEQ_LEN, SEQ_LEN)


def test_convert_embedding():
    embedding = ConveRTEmbedding(model_config)
    model_input = torch.randint(high=model_config.vocab_size, size=(train_config.batch_size, SEQ_LEN))
    embedding_output = embedding.forward(input_ids=model_input)

    assert embedding_output.size() == (train_config.batch_size, SEQ_LEN, model_config.num_embed_hidden)


def test_convert_encoder_layer():
    embedding = ConveRTEmbedding(model_config)
    encoder_layer = ConveRTEncoderLayer(model_config)

    model_input = torch.randint(high=model_config.vocab_size, size=(train_config.batch_size, SEQ_LEN))
    embedding_output = embedding.forward(input_ids=model_input)
    encoder_layer_output = encoder_layer.forward(embedding_output)

    assert encoder_layer_output.size() == (train_config.batch_size, SEQ_LEN, model_config.num_embed_hidden)


def test_convert_shared_encoder():
    shared_encoder = ConveRTSharedEncoder(model_config)

    input_ids = torch.randint(high=model_config.vocab_size, size=(train_config.batch_size, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(train_config.batch_size)])
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=None
    )

    encoder_output = shared_encoder(encoder_input)

    assert encoder_output.size() == (train_config.batch_size, SEQ_LEN, model_config.num_embed_hidden)


def test_convert_encoder():
    shared_encoder = ConveRTSharedEncoder(model_config)
    encoder = ConveRTEncoder(model_config, shared_encoder)

    input_ids = torch.randint(high=model_config.vocab_size, size=(train_config.batch_size, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(train_config.batch_size)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(train_config.batch_size)], dtype=torch.float)
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    encoder_output = encoder.forward(encoder_input)

    assert encoder_output.size() == (train_config.batch_size, model_config.feed_forward2_hidden)


def test_convert_dual_encoder():
    dual_encoder = ConveRTDualEncoder(model_config)

    input_ids = torch.randint(high=model_config.vocab_size, size=(train_config.batch_size, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(train_config.batch_size)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(train_config.batch_size)], dtype=torch.float)
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    context_embed, reply_embed = dual_encoder(encoder_input, encoder_input)

    assert context_embed.size() == (train_config.batch_size, model_config.feed_forward2_hidden)
    assert reply_embed.size() == (train_config.batch_size, model_config.feed_forward2_hidden)


def test_convert_dual_encoder_loss():
    dual_encoder = ConveRTDualEncoder(model_config)
    criterion = ConveRTCosineLoss(model_config)

    input_ids = torch.randint(high=model_config.vocab_size, size=(train_config.batch_size, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(train_config.batch_size)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(train_config.batch_size)], dtype=torch.float)
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    context_embed, reply_embed = dual_encoder(encoder_input, encoder_input)
    similarity = criterion.calculate_similarity(context_embed, reply_embed)
    loss, correct, total = criterion.calculate_loss(similarity)

    assert isinstance(loss.item(), float)
    assert isinstance(correct, int)
    assert isinstance(total, int)

    assert loss > 0


def test_convert_dual_encoder_splited_loss():
    dual_encoder = ConveRTDualEncoder(model_config)
    criterion = ConveRTCosineLoss(split_size=train_config.split_size)

    input_ids = torch.randint(high=model_config.vocab_size, size=(train_config.batch_size, SEQ_LEN))
    position_ids = torch.tensor([[i for i in range(SEQ_LEN)] for _ in range(train_config.batch_size)])
    input_lengths = torch.tensor([SEQ_LEN for _ in range(train_config.batch_size)], dtype=torch.float)
    encoder_input = ConveRTEncoderInput(
        input_ids=input_ids, position_ids=position_ids, attention_mask=None, input_lengths=input_lengths
    )

    context_embed, reply_embed = dual_encoder(encoder_input, encoder_input)
    similarity = criterion.calculate_similarity(context_embed, reply_embed, split_size=train_config.split_size)
    assert similarity.size() == (
        train_config.batch_size // train_config.split_size,
        train_config.split_size,
        train_config.split_size,
    )

    loss, correct, total = criterion.calculate_loss(similarity)

    assert isinstance(loss.item(), float)
    assert isinstance(correct, int)
    assert isinstance(total, int)

    assert loss > 0


def test_context_reply_multiplication():
    embedding = torch.rand(train_config.batch_size, model_config.feed_forward2_hidden, dtype=torch.float)
    label = torch.arange(train_config.batch_size, dtype=torch.long)

    consine_sim = torch.matmul(embedding, embedding.transpose(-1, -2))
    output_argmax = consine_sim.argmax(-1)

    assert output_argmax.size()[0] == train_config.batch_size
    assert output_argmax.eq(label).all().item() is True


def test_context_reply_split_multiplication():
    batch_size = train_config.batch_size * train_config.split_size

    embedding = torch.rand(batch_size, model_config.feed_forward2_hidden, dtype=torch.float)
    label = torch.arange(train_config.split_size, dtype=torch.long).repeat(batch_size // train_config.split_size)

    split_embedding = embedding.view(batch_size // train_config.split_size, train_config.split_size, -1)
    split_cosine_sim = torch.matmul(split_embedding, split_embedding.transpose(-1, -2))

    reduced_cosine_sim = split_cosine_sim.view(batch_size, train_config.split_size)

    output_argmax = reduced_cosine_sim.argmax(-1)

    assert output_argmax.size()[0] == batch_size
    assert output_argmax.eq(label).all().item() is True
