import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AttentionDoubleLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=2):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.ln_proj = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        self.attention_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        e_t = F.relu(self.ln_proj(self.projection(x)))
        lstm_out, _ = self.lstm(e_t)
        lstm_out = self.ln_lstm(lstm_out)
        attn_scores = self.attention_fc(lstm_out) # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1) # [batch_size, seq_len, 1]
        c_ctx = torch.sum(attn_weights * lstm_out, dim=1) # [batch_size, hidden_dim]
        return c_ctx

class PPOAgent(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128):
        super().__init__()
        self.feature_extractor = AttentionDoubleLSTM(input_dim, hidden_dim)
        
        # Nhánh backbone độc lập cho Critic và Actor để chuyên môn hóa đặc trưng
        self.critic_backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.actor_backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.critic = nn.Linear(hidden_dim, 1)
        self.actor_targ = nn.Linear(hidden_dim, 4)
        self.actor_lr   = nn.Linear(hidden_dim, 3)
        self.actor_mult = nn.Linear(hidden_dim, 3)
        self.actor_enh  = nn.Linear(hidden_dim, 3)

    def forward(self, state):
        c_ctx = self.feature_extractor(state)
        
        # Nhánh tính toán Value
        val_features = self.critic_backbone(c_ctx)
        value = self.critic(val_features)
        
        # Nhánh tính toán Policy
        act_features = self.actor_backbone(c_ctx)
        logits_list = [
            self.actor_targ(act_features),
            self.actor_lr(act_features),
            self.actor_mult(act_features),
            self.actor_enh(act_features)
        ]
        return value, logits_list

    def get_action(self, state, action=None):
        value, logits_list = self.forward(state)
        actions, log_probs, entropies = [], [], []
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            act = dist.sample() if action is None else action[:, i]
            actions.append(act)
            log_probs.append(dist.log_prob(act))
            entropies.append(dist.entropy())
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1).sum(dim=-1), entropies, value
