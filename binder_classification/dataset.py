import os
import sys
import config
import torch as T
import pandas as pd

from antiberty import AntiBERTyRunner
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

ANTIBERTY = AntiBERTyRunner()

def create_train_valid_sets(test_size: int = 0.2) -> None:

    dataset_path = os.path.join("./dataset", f"{config.DATASET_FILENAME}.csv")

    if not os.path.exists(dataset_path):
        print(f"\nThe dataset file '{dataset_path}.csv' is not exist.")
        sys.exit(1)
    
    df = pd.read_csv(dataset_path)

    required_cols = {"v_Lc", "link", "v_Rc", "label"}
    if not required_cols.issubset(df.columns):
        print(f"\nMissing required column(s): {required_cols - set(df.columns)}.")
        sys.exit(1)

    x, y = df.drop(columns=["label"]), df["label"]

    x_train, x_valid, y_train, y_valid = \
        train_test_split(x, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=y)
    
    x_train: pd.DataFrame
    train_df = x_train.copy()
    train_df["label"] = y_train

    x_valid: pd.DataFrame
    valid_df = x_valid.copy()
    valid_df["label"] = y_valid

    save_dir = os.path.join("./binder_classification", "logs")
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "train.csv")
    valid_path = os.path.join(save_dir, "valid.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

def get_train_batches(batch_size: int):

    train_df = pd.read_csv(os.path.join("./binder_classification", "logs", "train.csv"))
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    for L_idx in range(0, len(train_df), batch_size):

        batch_df = train_df.iloc[L_idx:L_idx + batch_size]

        v_Lcs = batch_df["v_Lc"].tolist()
        links = batch_df["link"].tolist()
        v_Rcs = batch_df["v_Rc"].tolist()

        v_Lc_embs, link_embs, v_Rc_embs = get_scFv_embs(v_Lcs, links, v_Rcs)

        labels  = T.tensor(batch_df["label"].values, dtype=T.float32)

        yield v_Lc_embs, link_embs, v_Rc_embs, labels

def get_valid_batches(batch_size: int):

    valid_df = pd.read_csv(os.path.join("./binder_classification", "logs", "valid.csv"))

    for L_idx in range(0, len(valid_df), batch_size):

        batch_df = valid_df.iloc[L_idx:L_idx + batch_size]

        v_Lcs = batch_df["v_Lc"].tolist()
        links = batch_df["link"].tolist()
        v_Rcs = batch_df["v_Rc"].tolist()

        v_Lc_embs, link_embs, v_Rc_embs = get_scFv_embs(v_Lcs, links, v_Rcs)

        labels  = T.tensor(batch_df["label"].values, dtype=T.float32)

        yield v_Lc_embs, link_embs, v_Rc_embs, labels

def get_scFv_embs(v_Lcs: list[str], links: list[str], v_Rcs: list[str]) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
    
    v_Lc_embs: list[T.Tensor] = [emb[1:-1] for emb in ANTIBERTY.embed(v_Lcs)]
    link_embs: list[T.Tensor] = [emb[1:-1] for emb in ANTIBERTY.embed(links)]
    v_Rc_embs: list[T.Tensor] = [emb[1:-1] for emb in ANTIBERTY.embed(v_Rcs)]

    v_Lc_embs: T.Tensor = pad_sequence(v_Lc_embs, batch_first=True)
    link_embs: T.Tensor = pad_sequence(link_embs, batch_first=True)
    v_Rc_embs: T.Tensor = pad_sequence(v_Rc_embs, batch_first=True)
    
    return v_Lc_embs, link_embs, v_Rc_embs