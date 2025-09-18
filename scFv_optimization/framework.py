import os
import config
import torch as T
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scFv_optimization.ppo import PPO
from scFv_optimization._utils import *
from scFv_optimization.environment import Environment
    
class Framework:

    def __init__(self) -> None:
        
        set_seeds()
        self.save_dir = get_save_dir()
        save_config(config, self.save_dir)

        self.env = Environment()

        self.state_dim = self.env.state_dim
        self.n_action1 = self.env.n_action1
        self.n_action2 = self.env.n_action2

        self.agent = PPO(self.state_dim, self.n_action1, self.n_action2, self.save_dir)

        self.exp_results_df = pd.DataFrame(columns=["Episode", "v_Lc_T", "link_T", "v_Rc_T", "Bind-Prob_T"])

    def train(self) -> None:

        with tqdm(total=config.PPO_N_EPISODES, desc="Training scFv Optimizer", unit="episode") as bar:

            self.episode = 0
            while self.episode < config.PPO_N_EPISODES:

                self.trjs = {"states": [], "action1s": [], "action2s": [], "rewards": [], 
                             "log_prob1s": [], "log_prob2s": [], "pred_values": []}

                states = self.env.reset()
                while True:

                    action1s, action2s, log_prob1s, log_prob2s, pred_values = self.agent.choose_actions(states)
                    next_states, rewards, done = self.env.step(action1s, action2s)

                    self._update_trjs(states, action1s, action2s, rewards, log_prob1s, log_prob2s, pred_values)
                    
                    if done:
                        self.agent.buffer.store_trjs(self.trjs)

                        if len(self.agent.buffer.data["states"]) >= config.PPO_BUFFER_SIZE:
                            self.agent.learn()

                        break
                    else:
                        states = next_states.clone()

                self.episode += config.N_PARALLELS
                bar.update(config.N_PARALLELS)
                
                self._update_exp_results_df()

                if (self.episode % config.CHECKPOINT_INTERVAL == 0) or (self.episode == config.PPO_N_EPISODES):
                    self.exp_results_df.to_csv(os.path.join(self.save_dir, "exp_results.csv"), index=False)

                    self._plot_bind_prob_curve()

    def _update_trjs(self, states: T.Tensor, action1s: T.Tensor, action2s: T.Tensor, rewards: T.Tensor, 
                     log_prob1s: T.Tensor, log_prob2s: T.Tensor, pred_values: T.Tensor) -> None:

        self.trjs["states"].append(states)
        self.trjs["action1s"].append(action1s)
        self.trjs["action2s"].append(action2s)
        self.trjs["rewards"].append(rewards)

        self.trjs["log_prob1s"].append(log_prob1s)
        self.trjs["log_prob2s"].append(log_prob2s)
        self.trjs["pred_values"].append(pred_values)

    def _update_exp_results_df(self) -> None:
        
        scFvs_T = self.env.scFvs_T
        v_Lc_len = len(config.TARGET_V_LC)
        link_len = len(config.TARGET_LINK)

        v_Lcs = [s[:v_Lc_len] for s in scFvs_T]
        links = [s[v_Lc_len:v_Lc_len + link_len] for s in scFvs_T]
        v_Rcs = [s[v_Lc_len + link_len:] for s in scFvs_T]

        bind_probs_T = self.env.bind_probs_curr.cpu().tolist()

        new_rows = []
        for n in range(config.N_PARALLELS):
            episode = self.episode - config.N_PARALLELS + n + 1
            
            new_rows.append({"Episode": f"{episode:06d}", 
                             "v_Lc_T": v_Lcs[n], "link_T": links[n], "v_Rc_T": v_Rcs[n], 
                             "Bind-Prob_T": f"{bind_probs_T[n]:.4f}"})

        if len(self.exp_results_df) == 0:
            self.exp_results_df = pd.DataFrame(new_rows)
        else:
            self.exp_results_df = pd.concat([
                self.exp_results_df, pd.DataFrame(new_rows)], ignore_index=True)

    def _plot_bind_prob_curve(self, sigma: int = 200) -> None:

        y_data = self.exp_results_df["Bind-Prob_T"].astype(float).to_list()
        x_data = range(1, len(y_data) + 1)
        
        y_data_smooth = gaussian_filter1d(y_data, sigma=sigma)

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data_smooth)

        plt.xlabel("Episodes", labelpad=10)
        plt.ylabel("Binding Probability", labelpad=10)

        plt.title("Binding Probability Across Episodes", pad=10)
        plt.grid(True)

        fig_path = os.path.join(self.save_dir, "bind_prob_curve.svg")
        plt.savefig(fig_path)
        plt.close()
