import transformers
import evaluate

from intentpl.utils import config
from intentpl.utils import data_utils
from intentpl.eval import eval_utils


if __name__ == "__main__":
    CONFIG = config.Config()

    dataset = data_utils.get_dataset_for_classification()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONFIG.HERBERT_MODEL_NAME)
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True),
        batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    num_labels = len(set(dataset["train"]["label"]))
    class_label = dataset["train"].features["label"]
    id2label = {
        i: label
        for label, i in zip(
            class_label.names,
            (class_label.str2int(name) for name in class_label.names))
    }
    label2id = {v: k for k, v in id2label.items()}

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        CONFIG.HERBERT_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    training_args = transformers.TrainingArguments(
        output_dir=CONFIG.TRAINING_OUTPUT_DIR,
        learning_rate=CONFIG.HERBERT_FINETUNING_LR,
        per_device_train_batch_size=CONFIG.HERBERT_PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=CONFIG.HERBERT_PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=CONFIG.HERBERT_NUM_EPOCHS,
        weight_decay=CONFIG.HERBERT_WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=eval_utils.compute_metrics(metric=accuracy),
    )
    trainer.train()
