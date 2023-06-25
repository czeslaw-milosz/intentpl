from dataclasses import dataclass, field

from typing import List


@dataclass
class Config:
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

    HERBERT_MODEL_NAME: str = "allegro/herbert-large-cased"
    HERBERT_FINETUNING_LR: float = 2e-5
    HERBERT_PER_DEVICE_TRAIN_BATCH_SIZE: int = 16
    HERBERT_PER_DEVICE_EVAL_BATCH_SIZE: int = 16
    HERBERT_NUM_EPOCHS: int = 2
    HERBERT_WEIGHT_DECAY: float = 0.01


    TRAINING_OUTPUT_DIR: str = "../../resources"
