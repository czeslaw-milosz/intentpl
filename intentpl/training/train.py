import functools

import transformers
import evaluate

from intentpl.utils import config
from intentpl.utils import data_utils
from intentpl.eval import eval_utils


def run_train(params):
    model_name = params["model_name"]    
    training_args = params["training_args"]

    dataset = data_utils.get_dataset_for_classification()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True),
        batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

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
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=eval_utils.compute_metrics,
    )
    
    trainer.train()
