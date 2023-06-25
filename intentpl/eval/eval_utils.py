import evaluate
import numpy as np


def compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
