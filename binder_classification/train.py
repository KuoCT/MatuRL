import os
import torch as T
import numpy as np

from binder_classification._utils import set_seeds
from binder_classification.dataset import create_train_valid_sets
from binder_classification.dataset import get_train_batches, get_valid_batches
from binder_classification.classifier import BinderClassifier

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

def train(batch_size: int = 128) -> None:

    set_seeds()
    create_train_valid_sets()

    model = BinderClassifier(os.path.join("./binder_classification", "logs", "model.pt"))

    best_valid_metrics = {"acc": 0, "rec": 0, "spe": 0, "pre": 0,
                          "f1s": 0, "mcc": 0, "auc": 0}

    print("\nStart training binder classifier ...")

    early_stop_count, patient = 0, 10
    for epoch in range(1, 101):
        model.train()

        for v_Lc_embs, link_embs, v_Rc_embs, labels in get_train_batches(batch_size):
            model.optimizer.zero_grad()

            logits: T.Tensor = model(v_Lc_embs, link_embs, v_Rc_embs)

            logits = logits.squeeze(1)
            labels = labels.to(DEVICE)

            loss: T.Tensor = model.criterion(logits, labels)

            loss.backward()
            model.optimizer.step()
        
        model.scheduler.step()
        
        valid_metrics = valid(model, batch_size)

        if (valid_metrics["mcc"] >= best_valid_metrics["mcc"]) and \
           (valid_metrics["auc"] >= best_valid_metrics["auc"]):
            
            print(f"epoch: [{epoch:003d}] | "
                  f"valid_acc: {valid_metrics['acc']:.4f} | valid_rec: {valid_metrics['rec']:.4f} | "
                  f"valid_spe: {valid_metrics['spe']:.4f} | valid_pre: {valid_metrics['pre']:.4f} | "
                  f"valid_f1s: {valid_metrics['f1s']:.4f} | valid_mcc: {valid_metrics['mcc']:.4f} | "
                  f"valid_auc: {valid_metrics['auc']:.4f}")
            
            best_valid_metrics = valid_metrics
            early_stop_count = 0
            model.save_model()

        else:
            early_stop_count += 1
            if early_stop_count == patient:
                print(f"\nEarly stopping at epoch {epoch:003d}.")
                break

def valid(model: BinderClassifier, batch_size: int = 128) -> dict[str, float]:

    y_true_batches: list[np.ndarray] = []
    y_prob_batches: list[np.ndarray] = []

    model.eval()
    with T.no_grad():
        for v_Lc_embs, link_embs, v_Rc_embs, labels in get_valid_batches(batch_size):

            logits: T.Tensor = model(v_Lc_embs, link_embs, v_Rc_embs)

            logits = logits.squeeze(1)
            labels = labels.to(DEVICE)

            y_true_batches.append(labels.cpu().numpy())
            y_prob_batches.append(T.sigmoid(logits).cpu().numpy())

    y_trues = np.concatenate(y_true_batches, axis=0)
    y_probs = np.concatenate(y_prob_batches, axis=0)
    y_preds = (y_probs >= 0.5).astype(np.int32)

    acc = accuracy_score(y_trues, y_preds)
    rec = recall_score(y_trues, y_preds)

    tn, fp, _, _ = confusion_matrix(y_trues, y_preds, labels=[0, 1]).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    pre = precision_score(y_trues, y_preds, zero_division=0)

    f1s = f1_score(y_trues, y_preds)
    mcc = matthews_corrcoef(y_trues, y_preds)

    auc = roc_auc_score(y_trues, y_probs)

    return {"acc": acc, "rec": rec, "spe": spe, "pre": pre,
            "f1s": f1s, "mcc": mcc, "auc": auc}
