import transformers
from dataclasses import dataclass, field

from typing import List


@dataclass
class DataConfig:
    MASSIVE_DATASET_NAME: str = "AmazonScience/massive"
    MASSIVE_LOCALE: str = "pl-PL"
    MASSIVE_UNUSED_COLUMNS: List[str] = field(default_factory = lambda: [
        "locale",
        "scenario",
        "annot_utt",
        "worker_id",
        "slot_method",
        "judgments"
    ])
    TRAINING_OUTPUT_DIR: str = "../../resources"

HERBERT_ARGS = {
    "model_name": "allegro/herbert-base-cased",
    "training_args": transformers.TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        output_dir=DataConfig.TRAINING_OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
}

#TODO finetune hyperparameters for mt5
MT5_ARGS = {
    "model_name": "google/mt5-base",
    "training_args": transformers.TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        output_dir=DataConfig.TRAINING_OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
}

# TODO finetune hyperparameters for xlmr
XLMR_ARGS = {
    "model_name": "xlm-roberta-base",
    "training_args": transformers.TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        output_dir=DataConfig.TRAINING_OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
}
