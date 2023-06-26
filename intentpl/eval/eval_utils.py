import evaluate
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    return {
        "accuracy": accuracy.compute(
            predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(
            predictions=predictions, references=labels, average="macro"
        )
    }
    # return metric_fun.compute(predictions=predictions, references=labels)
