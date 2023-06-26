from intentpl.utils.config import HERBERT_ARGS, MT5_ARGS, XLMR_ARGS
from intentpl.utils import data_utils
from intentpl.eval import eval_utils
from intentpl.training import train


if __name__ == "__main__":
    train.run_train(XLMR_ARGS)
