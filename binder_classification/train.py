import config
import torch as T
import numpy as np

from device import DEVICE
from binder_classification._utils import set_seeds
from binder_classification.dataset import create_train_valid_sets
from binder_classification.dataset import get_train_batches, get_valid_batches
from binder_classification.classifier import BinderClassifier

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

def train(batch_size: int = 128) -> None:

    set_seeds()
    create_train_valid_sets()

    config.BINDER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = BinderClassifier(str(config.BINDER_MODEL_PATH))

    best_valid_metrics = {"acc": 0, "rec": 0, "spe": 0, "pre": 0,
                          "f1s": 0, "mcc": 0, "auc": 0}

    print("Start training binder classifier ...")

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

        is_progress = (
            valid_metrics["mcc"] >= best_valid_metrics["mcc"]
            and valid_metrics["auc"] >= best_valid_metrics["auc"]
        )
        progress_label = "[Progress]" if is_progress else ""

        print(
            f"Epoch: {epoch:03d} {progress_label:<10} Validate: "
            f"ACC={valid_metrics['acc']:.4f} | REC={valid_metrics['rec']:.4f} | "
            f"SPE={valid_metrics['spe']:.4f} | PRE={valid_metrics['pre']:.4f} | "
            f"F1S={valid_metrics['f1s']:.4f} | MCC={valid_metrics['mcc']:.4f} | "
            f"AUC={valid_metrics['auc']:.4f}"
        )

        if is_progress:
            best_valid_metrics = valid_metrics
            early_stop_count = 0
            model.save_model()

        else:
            early_stop_count += 1
            if early_stop_count == patient:
                print(f"Early stopping at epoch {epoch:003d}.")
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
