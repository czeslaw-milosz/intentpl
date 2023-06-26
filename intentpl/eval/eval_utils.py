import evaluate
import numpy as np


def compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric_fun = evaluate.load(metric)
    return metric_fun.compute(predictions=predictions, references=labels)
