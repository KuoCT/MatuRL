import os
import config
import torch as T
import torch.nn.functional as F

from antiberty import AntiBERTyRunner
from binder_classification.train import train
from binder_classification.classifier import BinderClassifier

DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

class Environment:

    def __init__(self) -> None:

        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.a2_to_aa = {idx: aa for idx, aa in enumerate(self.amino_acids)}
        self.aa_to_a2 = {aa: idx for idx, aa in enumerate(self.amino_acids)}

        self.antiberty = AntiBERTyRunner()
        self._load_binder_classifier()

        self.scFvs_1 = [config.TARGET_V_LC + config.TARGET_LINK + config.TARGET_V_RC] * config.N_PARALLELS

        self.states_1: T.Tensor = self._get_one_hot_encoding(self.scFvs_1)
        self.state_dim = self.states_1.shape[1]

        self.n_action1 = len(self.scFvs_1[0])
        self.n_action2 = len(self.amino_acids)

        self.bind_probs_1 = self.get_bind_probs(self.scFvs_1)

    def _load_binder_classifier(self) -> None:

        save_path = os.path.join("./binder_classification", "logs", "model.pt")

        if not os.path.exists(save_path): train()

        self.binder_classifier = BinderClassifier(save_path)
        self.binder_classifier.load_model()

    def _get_one_hot_encoding(self, seqs: list[str]) -> T.Tensor:

        indices = T.tensor([[self.aa_to_a2[aa] for aa in seq] for seq in seqs], dtype=T.long)
        one_hot = F.one_hot(indices.to(DEVICE), num_classes=len(self.amino_acids))

        return one_hot.view(len(seqs), -1).to(T.float32)

    def get_bind_probs(self, scFvs: list[str]) -> T.Tensor:

        v_Lc_len = len(config.TARGET_V_LC)
        link_len = len(config.TARGET_LINK)

        v_Lcs = [s[:v_Lc_len] for s in scFvs]
        links = [s[v_Lc_len:v_Lc_len + link_len] for s in scFvs]
        v_Rcs = [s[v_Lc_len + link_len:] for s in scFvs]

        v_Lc_embeds = T.stack(self.antiberty.embed(v_Lcs))[:, 1:-1, :]
        link_embeds = T.stack(self.antiberty.embed(links))[:, 1:-1, :]
        v_Rc_embeds = T.stack(self.antiberty.embed(v_Rcs))[:, 1:-1, :]

        with T.no_grad():
            self.binder_classifier.eval()
            bind_probs: T.Tensor = self.binder_classifier(v_Lc_embeds, link_embeds, v_Rc_embeds)
            bind_probs = T.sigmoid(bind_probs).squeeze(-1)

        return bind_probs

    def reset(self) -> T.Tensor:

        self.done = False
        self.time_step = 1

        self.scFvs_curr = self.scFvs_1.copy()
        self.scFvs_prev = self.scFvs_1.copy()

        self.bind_probs_curr = self.bind_probs_1.clone()
        self.bind_probs_prev = self.bind_probs_1.clone()

        return self.states_1

    def step(self, action1s: T.Tensor, action2s: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:

        action1s = action1s.tolist()
        action2s = action2s.tolist()

        self.scFvs_prev = self.scFvs_curr.copy()
        new_aas = [self.a2_to_aa[a2] for a2 in action2s]
        self.scFvs_curr = [p[:a1] + aa + p[a1 + 1:] for p, a1, aa 
                           in zip(self.scFvs_curr, action1s, new_aas)]

        self.bind_probs_prev.copy_(self.bind_probs_curr)
        self.bind_probs_curr = self.get_bind_probs(self.scFvs_curr)

        if self.time_step == config.PPO_TIME_HORIZON:
            self.scFvs_T = self.scFvs_curr.copy()
            self.done = True
        else:
            self.time_step += 1

        return self._get_one_hot_encoding(self.scFvs_curr), self._get_rewards(), self.done

    def _get_rewards(self) -> T.Tensor:

        bind_prob_diffs = self.bind_probs_curr - self.bind_probs_prev

        if self.done:
            bind_prob_T_diffs = self.bind_probs_curr - self.bind_probs_1
            factors = T.where(bind_prob_T_diffs > 0, 1 - self.bind_probs_curr, self.bind_probs_curr)
            rewards = bind_prob_diffs + (bind_prob_T_diffs / T.clamp(factors, min=1e-2))
        else:
            rewards = bind_prob_diffs

        return rewards.cpu()
