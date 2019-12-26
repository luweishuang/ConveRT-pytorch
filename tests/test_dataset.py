import json

from torch.utils.data import DataLoader

from convert.dataset import ConveRTDataConfig, ConveRTDataset, ConveRTExample, ConveRTTextUtility

REDDIT_SAMPLE_DATA = {
    "context_author": "Needs_Mega_Magikarp",
    "context/8": "*She giggles at his giggle.* Yay~",
    "context/5": "*He rests his head on yours.*\n\nYou aaaaare. You're the cutest.",
    "context/4": "Pfffft. *She playfully pokes his stomach.* Shuddup.",
    "context/7": "*He hugs you.*\n\nOhmigooods, you're so cute.",
    "context/6": "*She giggles again.* No I'm noooot.",
    "context/1": "*He snorts a laugh*\n\nD'aww. Cute.",
    "context/0": "Meanie.",
    "response_author": "Ironic_Remorse",
    "subreddit": "PercyJacksonRP",
    "thread_id": "2vcitx",
    "context/3": "*He shrugs.*\n\nBut I dun wanna lie!",
    "context": "Cutie.\n\n*He jokes, rubbing your arm again. Vote Craig for best brother 2k15.*",
    "context/2": "*She sticks her tongue out.*",
    "response": "Meanieee. *She pouts.*",
}


def test_load_reddit_example():
    example = ConveRTExample.load_reddit_json(REDDIT_SAMPLE_DATA)
    assert example.response == "Meanieee. *She pouts.*"
    target_context = [
        "Cutie.\n\n*He jokes, rubbing your arm again. Vote Craig for best brother 2k15.*",
        "Meanie.",
        "*He snorts a laugh*\n\nD'aww. Cute.",
        "*She sticks her tongue out.*",
        "*He shrugs.*\n\nBut I dun wanna lie!",
        "Pfffft. *She playfully pokes his stomach.* Shuddup.",
        "*He rests his head on yours.*\n\nYou aaaaare. You're the cutest.",
        "*She giggles again.* No I'm noooot.",
        "*He hugs you.*\n\nOhmigooods, you're so cute.",
        "*She giggles at his giggle.* Yay~",
    ]

    for source, target in zip(example.context, target_context):
        assert source == target


def test_load_reddit_examples():
    with open("data/sample-dataset.json") as f:
        examples = [ConveRTExample.load_reddit_json(json.loads(line.strip())) for line in f]

    assert len(examples) == 1000


def test_loading_sp_model():
    config = ConveRTDataConfig(
        sp_model_path="data/en.wiki.bpe.vs10000.model", train_dataset_dir=None, test_dataset_dir=None
    )
    tokenizer = ConveRTTextUtility._load_tokenizer(config)
    assert tokenizer is not None


def test_encoding_using_sp_model():
    config = ConveRTDataConfig(
        sp_model_path="data/en.wiki.bpe.vs10000.model", train_dataset_dir=None, test_dataset_dir=None
    )
    tokenizer = ConveRTTextUtility(config)
    assert tokenizer.encode_string_to_tokens("welcome home") == [3441, 4984, 1004]


def test_dataset_get_item():
    config = ConveRTDataConfig(
        sp_model_path="data/en.wiki.bpe.vs10000.model", train_dataset_dir=None, test_dataset_dir=None
    )
    text_utility = ConveRTTextUtility(config)
    examples = [ConveRTExample.load_reddit_json(REDDIT_SAMPLE_DATA)] * 10
    dataset = ConveRTDataset(examples, text_utility)

    assert len(dataset) == 10


def test_dataset_batching():
    config = ConveRTDataConfig(
        sp_model_path="data/en.wiki.bpe.vs10000.model", train_dataset_dir=None, test_dataset_dir=None
    )
    text_utility = ConveRTTextUtility(config)
    examples = [ConveRTExample.load_reddit_json(REDDIT_SAMPLE_DATA)] * 10
    dataset = ConveRTDataset(examples, text_utility)
    data_loader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)

    for batch in data_loader:
        print(batch)
