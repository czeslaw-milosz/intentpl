import datasets
from datasets.dataset_dict import DatasetDict

from intentpl.utils.config import Config


CONFIG = Config()


def get_dataset_for_classification(
        dataset_name: str = CONFIG.MASSIVE_DATASET_NAME,
        locale: str = CONFIG.MASSIVE_LOCALE) -> DatasetDict:
    dataset = datasets.load_dataset(dataset_name, locale)
    for subset_name in dataset:
        dataset[subset_name] = dataset[subset_name].remove_columns(
            CONFIG.MASSIVE_UNUSED_COLUMNS)
        dataset[subset_name] = dataset[subset_name].rename_column(
            "intent", "label")
        dataset[subset_name] = dataset[subset_name].rename_column(
            "utt", "text")
    return dataset


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)
