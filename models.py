import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AttentionDoubleLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=2):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.attention_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        e_t = F.relu(self.projection(x))
        lstm_out, _ = self.lstm(e_t)
        attn_scores = self.attention_fc(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        c_ctx = torch.sum(attn_weights * lstm_out, dim=1)
        return c_ctx

class PPOAgent(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128):
        super().__init__()
        self.feature_extractor = AttentionDoubleLSTM(input_dim, hidden_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.actor_targ = nn.Linear(hidden_dim, 4)
        self.actor_lr   = nn.Linear(hidden_dim, 3)
        self.actor_mult = nn.Linear(hidden_dim, 3)
        self.actor_enh  = nn.Linear(hidden_dim, 3)

    def forward(self, state):
        c_ctx = self.feature_extractor(state)
        value = self.critic(c_ctx)
        return value, [self.actor_targ(c_ctx), self.actor_lr(c_ctx), self.actor_mult(c_ctx), self.actor_enh(c_ctx)]

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
