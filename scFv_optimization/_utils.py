import os
import json
import types
import config
import random
import torch as T
import numpy as np

from datetime import datetime

def set_seeds() -> None:

    random.seed(config.RANDOM_SEED)
    T.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if T.cuda.is_available():
        T.cuda.manual_seed(config.RANDOM_SEED)
        T.cuda.manual_seed_all(config.RANDOM_SEED)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False

def get_save_dir() -> str:

    today = datetime.today().strftime("%Y-%m-%d")
    logs_dir = os.path.join("./scFv_optimization", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    prefix = f"{today}-"
    existing = [d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith(prefix)]

    nums = [int(d.split("-")[-1]) for d in existing 
            if d.split("-")[-1].isdigit()]
    next_num = max(nums, default=0) + 1

    dir_name = f"{today}-{next_num:02d}"
    save_dir = os.path.join(logs_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_config(module: types.ModuleType, save_dir: str):

    config_dict = {k: v for k, v in vars(module).items() 
                   if not k.startswith("__") and not callable(v)}
    
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

