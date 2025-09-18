import config
import random
import torch as T
import numpy as np

def set_seeds() -> None:

    random.seed(config.RANDOM_SEED)
    T.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if T.cuda.is_available():
        T.cuda.manual_seed(config.RANDOM_SEED)
        T.cuda.manual_seed_all(config.RANDOM_SEED)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False