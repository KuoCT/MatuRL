import config
import torch as T
import torch.nn as nn
import torch.optim as optim

DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

class Actor1(nn.Module):

    def __init__(self, state_dim: int, n_action1: int, pt_file_path: str) -> None:

        super(Actor1, self).__init__()

        self.pt_file_path = pt_file_path

        self.policy_net = nn.Sequential(nn.Linear(state_dim, config.AGENTS_HIDDEN_DIM),
                                        nn.LayerNorm(config.AGENTS_HIDDEN_DIM),
                                        nn.ReLU(),
                                        nn.Dropout(config.AGENTS_DROPOUT_RATE),
                                        nn.Linear(config.AGENTS_HIDDEN_DIM, config.AGENTS_HIDDEN_DIM),
                                        nn.LayerNorm(config.AGENTS_HIDDEN_DIM),
                                        nn.ReLU(),
                                        nn.Linear(config.AGENTS_HIDDEN_DIM, n_action1))

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=config.AGENTS_LR, weight_decay=1e-2)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   gamma=config.AGENTS_GAMMA, step_size=config.AGENTS_STEP_SIZE)
        self.to(DEVICE)

    def save_model(self) -> None:
        T.save(self.state_dict(), self.pt_file_path)

    def load_model(self) -> None:
        self.load_state_dict(T.load(self.pt_file_path, map_location=DEVICE, weights_only=False))

    def forward(self, states: T.Tensor) -> T.Tensor:
        return T.softmax(self.policy_net(states), dim=-1)
    
class Actor2(nn.Module):

    def __init__(self, state_dim: int, n_action1: int, n_action2: int, pt_file_path: str) -> None:

        super(Actor2, self).__init__()

        self.pt_file_path = pt_file_path

        self.pos_embed_dim = min(state_dim // 2, 160)
        self.pos_embed = nn.Embedding(n_action1, self.pos_embed_dim)

        self.policy_net = nn.Sequential(nn.Linear(state_dim + self.pos_embed_dim, config.AGENTS_HIDDEN_DIM),
                                        nn.LayerNorm(config.AGENTS_HIDDEN_DIM),
                                        nn.ReLU(),
                                        nn.Dropout(config.AGENTS_DROPOUT_RATE),
                                        nn.Linear(config.AGENTS_HIDDEN_DIM, config.AGENTS_HIDDEN_DIM),
                                        nn.LayerNorm(config.AGENTS_HIDDEN_DIM),
                                        nn.ReLU(),
                                        nn.Linear(config.AGENTS_HIDDEN_DIM, n_action2))

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=config.AGENTS_LR, weight_decay=1e-2)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   gamma=config.AGENTS_GAMMA, step_size=config.AGENTS_STEP_SIZE)
        self.to(DEVICE)

    def save_model(self) -> None:
        T.save(self.state_dict(), self.pt_file_path)

    def load_model(self) -> None:
        self.load_state_dict(T.load(self.pt_file_path, map_location=DEVICE, weights_only=False))

    def forward(self, states: T.Tensor, action1s: T.Tensor) -> T.Tensor:
        return T.softmax(self.policy_net(T.cat([states, self.pos_embed(action1s)], dim=-1)), dim=-1)

class Critic(nn.Module):

    def __init__(self, state_dim: int, pt_file_path: str) -> None:
        super(Critic, self).__init__()

        self.pt_file_path = pt_file_path

        self.value_net = nn.Sequential(nn.Linear(state_dim, config.AGENTS_HIDDEN_DIM),
                                       nn.LayerNorm(config.AGENTS_HIDDEN_DIM),
                                       nn.ReLU(),
                                       nn.Dropout(config.AGENTS_DROPOUT_RATE),
                                       nn.Linear(config.AGENTS_HIDDEN_DIM, config.AGENTS_HIDDEN_DIM),
                                       nn.LayerNorm(config.AGENTS_HIDDEN_DIM),
                                       nn.ReLU(),
                                       nn.Linear(config.AGENTS_HIDDEN_DIM, 1))

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=config.AGENTS_LR, weight_decay=1e-2)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   gamma=config.AGENTS_GAMMA, step_size=config.AGENTS_STEP_SIZE)
        self.to(DEVICE)

    def save_model(self) -> None:
        T.save(self.state_dict(), self.pt_file_path)

    def load_model(self) -> None:
        self.load_state_dict(T.load(self.pt_file_path, map_location=DEVICE, weights_only=False))

    def forward(self, state: T.Tensor) -> T.Tensor:
        return self.value_net(state)