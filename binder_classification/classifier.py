import torch as T
import torch.nn as nn
import torch.optim as optim

DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

class BinderClassifier(nn.Module):

    def __init__(self, save_path: str) -> None:
        
        super(BinderClassifier, self).__init__()

        self.save_path = save_path

        self.mha = SingleQueryMHA()
        
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        
        self.to(DEVICE)

    def save_model(self) -> None:
        T.save(self.state_dict(), self.save_path)

    def load_model(self) -> None:
        self.load_state_dict(T.load(self.save_path, map_location=DEVICE, weights_only=False))

    def forward(self, v_Lc_embeds: T.Tensor, link_embeds: T.Tensor, v_Hc_embeds: T.Tensor) -> T.Tensor:

        v_Lc_pooled = self.mha(v_Lc_embeds.to(DEVICE))
        link_pooled = self.mha(link_embeds.to(DEVICE))
        v_Hc_pooled = self.mha(v_Hc_embeds.to(DEVICE))

        x = T.cat([v_Lc_pooled, link_pooled, v_Hc_pooled], dim=1)

        return self.classifier(x)
        
class SingleQueryMHA(nn.Module):

    def __init__(self) -> None:

        super(SingleQueryMHA, self).__init__()

        self.linear_proj = nn.Linear(512, 32)

        self.query = nn.Parameter(T.randn(1, 1, 32))

        self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=False)

    def forward(self, x: T.Tensor) -> T.Tensor:

        batch_size, _, _ = x.size()

        x_proj: T.Tensor = self.linear_proj(x)
        x_proj = x_proj.transpose(0, 1)

        q = self.query.expand(-1, batch_size, -1)

        attn_output, _ = self.mha(q, x_proj, x_proj)
        attn_output: T.Tensor

        return attn_output.squeeze(0)