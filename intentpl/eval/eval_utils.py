import evaluate
import numpy as np


def compute_metrics(eval_pred, metrics):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # metric_fun = evaluate.load(metric)
    return {
        metric_name: evaluate.load(metric_name).compute(
            predictions=predictions, references=labels)[metric_name]
        for metric_name in metrics
    }
    # return metric_fun.compute(predictions=predictions, references=labels)
