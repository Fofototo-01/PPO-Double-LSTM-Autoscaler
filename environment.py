import numpy as np
import torch
import pandas as pd
from collections import deque
import math

class KubernetesEnv:
    def __init__(self, csv_path, max_steps=500):
        self.max_steps = max_steps
        self.df = pd.read_csv(csv_path)
        self.total_rows = len(self.df)
        self.current_row = 0
        self.state_buffer = deque(maxlen=3)
        self.cpu_target = 50

    def reset(self):
        self.current_step = 0
        if self.current_row + self.max_steps >= self.total_rows: self.current_row = 0
        for _ in range(3): self.state_buffer.append(np.zeros(14))
        return self._get_tensor_state()

    def _get_metrics_from_csv(self):
        row_data = self.df.iloc[self.current_row]
        self.current_row += 1
        s_t = np.zeros(14)
        s_t[1] = float(row_data['value'])
        s_t[3] = self.cpu_target / 100.0
        return s_t

    def _get_tensor_state(self):
        return torch.FloatTensor(np.array(self.state_buffer)).unsqueeze(0)

    def step(self, actions):
        self.current_step += 1
        a_targ = actions[0][0].item()
        self.cpu_target = {0: 30, 1: 50, 2: 70, 3: 90}[a_targ]
        new_s_t = self._get_metrics_from_csv()
        self.state_buffer.append(new_s_t)
        u_cpu = new_s_t[1] * 100.0
        diff = abs(u_cpu - self.cpu_target)
        reward = 1.0 if diff <= 10 else math.exp(-diff / 50.0)
        return self._get_tensor_state(), reward, self.current_step >= self.max_steps
