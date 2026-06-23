import numpy as np
import torch
import pandas as pd
from collections import deque
import math
import os

class KubernetesEnv:
    def __init__(self, csv_path, max_steps=500):
        self.max_steps = max_steps
        self.csv_path = csv_path
        
        # Cơ chế fallback tự động tạo dữ liệu nếu không tìm thấy file dataset của Colab
        if not os.path.exists(csv_path):
            print(f"[Warning] Dataset not found at '{csv_path}'. Generating synthetic CPU workload for local run...")
            # Tạo dữ liệu workload mô phỏng 20,000 bước thời gian (gồm dao động ngày đêm và nhiễu)
            np.random.seed(42)
            steps = 20000
            time = np.linspace(0, 40 * np.pi, steps)
            # Tải nền + Dao động tuần hoàn + Dao động đột biến nhỏ + Nhiễu Gaussian
            workload = 0.5 + 0.3 * np.sin(time) + 0.1 * np.sin(time / 4) + np.random.normal(0, 0.04, steps)
            workload = np.clip(workload, 0.05, 0.95)
            self.df = pd.DataFrame({'value': workload})
        else:
            self.df = pd.read_csv(csv_path)
            
        self.total_rows = len(self.df)
        self.current_row = 0
        self.state_buffer = deque(maxlen=3)
        
        # Các biến trạng thái của hạ tầng ảo hóa
        self.cpu_target = 50.0       # Ngưỡng CPU Target cấu hình (%)
        self.replicas = 3            # Số lượng Pod/VM hoạt động (1 -> 10)
        self.min_replicas = 1
        self.max_replicas = 10
        self.cooldown_remaining = 0  # Số bước còn lại trong thời gian chờ ổn định (cooldown)
        self.prev_replicas = 3       # Lưu số lượng replicas bước trước để tính dao động
        self.last_action_idx = [1, 0, 0, 0] # Lưu chỉ số hành động gần nhất

    def reset(self):
        self.current_step = 0
        self.replicas = 3
        self.cpu_target = 50.0
        self.cooldown_remaining = 0
        self.prev_replicas = 3
        self.last_action_idx = [1, 0, 0, 0]
        
        if self.current_row + self.max_steps >= self.total_rows: 
            self.current_row = 0
            
        # Làm sạch và điền trước bộ đệm 3 bước bằng vector 0
        self.state_buffer.clear()
        for _ in range(3): 
            self.state_buffer.append(np.zeros(14))
            
        return self._get_tensor_state()

    def _build_state_vector(self, workload_val):
        """
        Xây dựng vector trạng thái 14 chiều phong phú từ các thông số thực tế của cụm máy chủ:
        s_t[0]: Nhu cầu workload thô từ CSV (0.0 -> 1.0)
        s_t[1]: CPU utilization thực tế của hệ thống (workload / replicas) (0.0 -> 1.0)
        s_t[2]: Tỷ lệ số replica hiện tại (replicas / max_replicas)
        s_t[3]: Ngưỡng target CPU cài đặt (cpu_target / 100.0)
        s_t[4]: Hướng điều chỉnh tài nguyên ở bước trước (-1.0: giảm, 0.0: giữ nguyên, 1.0: tăng)
        s_t[5]: Tỷ lệ thời gian cooldown còn lại (cooldown_remaining / max_cooldown)
        s_t[6]: Tốc độ thay đổi CPU utilization so với bước trước
        s_t[7]: Bộ nhớ Ram giả lập (tỷ lệ thuận với CPU kèm nhiễu đo đạc)
        s_t[8]: Chỉ số độ trễ phản hồi (Response Time) (tăng phi tuyến theo lý thuyết hàng đợi M/M/1)
        s_t[9]: Trạng thái vi phạm SLA (1.0 nếu CPU thực tế > 90%, ngược lại 0.0)
        s_t[10]: Thành phần Sin của thời gian (biểu thị tính tuần hoàn trong ngày)
        s_t[11]: Thành phần Cos của thời gian
        s_t[12]: Lưu lượng QPS (truy vấn/giây) giả lập tỷ lệ thuận với workload
        s_t[13]: Lịch sử chỉ số hệ số nhân (multiplier) hành động trước
        """
        s_t = np.zeros(14)
        s_t[0] = workload_val
        
        # CPU thực tế tải trên các pod. Giả định 4.0 đơn vị workload là đầy tải cho 4 replicas
        actual_cpu = min(1.0, (workload_val * 4.0) / max(1.0, self.replicas))
        s_t[1] = actual_cpu
        s_t[2] = self.replicas / self.max_replicas
        s_t[3] = self.cpu_target / 100.0
        
        # Hướng scaling gần nhất
        if self.replicas > self.prev_replicas:
            s_t[4] = 1.0
        elif self.replicas < self.prev_replicas:
            s_t[4] = -1.0
        else:
            s_t[4] = 0.0
            
        s_t[5] = self.cooldown_remaining / 5.0 # Chuẩn hóa với cooldown tối đa = 5 bước
        
        # Tốc độ thay đổi CPU
        if len(self.state_buffer) > 0:
            s_t[6] = actual_cpu - self.state_buffer[-1][1]
        else:
            s_t[6] = 0.0
            
        # Bộ nhớ Ram giả lập (tải CPU cao thường kéo theo RAM cao)
        s_t[7] = min(0.95, workload_val * 0.75 + 0.15 + np.random.normal(0, 0.02))
        
        # Độ trễ phản hồi hệ thống (M/M/1 queue model: Latency = 1 / (1 - CPU))
        if actual_cpu >= 0.95:
            latency = 20.0
        else:
            latency = 1.0 / (1.0 - actual_cpu + 1e-5)
        s_t[8] = min(1.0, latency / 20.0) # chuẩn hóa về [0, 1]
        
        # Trạng thái SLA
        s_t[9] = 1.0 if actual_cpu > 0.90 else 0.0
        
        # Tín hiệu thời gian tuần hoàn
        s_t[10] = math.sin(self.current_step * 2 * math.pi / 100.0)
        s_t[11] = math.cos(self.current_step * 2 * math.pi / 100.0)
        s_t[12] = workload_val
        s_t[13] = self.last_action_idx[2] / 2.0 # Chuẩn hóa a_mult (0, 1, 2)
        
        return s_t

    def _get_tensor_state(self):
        return torch.FloatTensor(np.array(self.state_buffer)).unsqueeze(0)

    def step(self, actions):
        self.current_step += 1
        
        # Giải mã 4 hành động rời rạc từ Multi-Discrete Actor
        # actions shape: [batch, 4]
        a_targ = actions[0][0].item()
        a_lr   = actions[0][1].item()
        a_mult = actions[0][2].item()
        a_enh  = actions[0][3].item()
        
        self.last_action_idx = [a_targ, a_lr, a_mult, a_enh]
        
        # 1. Cấu hình Target CPU
        self.cpu_target = {0: 30.0, 1: 50.0, 2: 70.0, 3: 90.0}[a_targ]
        
        # 2. Đọc workload từ tập dữ liệu
        row_data = self.df.iloc[self.current_row]
        workload_val = float(row_data['value'])
        self.current_row += 1
        if self.current_row >= self.total_rows: 
            self.current_row = 0
            
        # 3. Tính toán CPU trước khi scaling
        current_cpu = min(1.0, (workload_val * 4.0) / max(1.0, self.replicas))
        
        # 4. Thuật toán HPA ước lượng replica mong muốn: Desired = ceil(Current * (Current_CPU / Target_CPU))
        target_ratio = self.cpu_target / 100.0
        desired_replicas = math.ceil(self.replicas * (current_cpu / target_ratio))
        desired_replicas = max(self.min_replicas, min(self.max_replicas, desired_replicas))
        
        self.prev_replicas = self.replicas
        
        # Quản lý cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            
        # 5. Thực thi logic scaling có ảnh hưởng bởi a_mult, a_lr, a_enh
        if desired_replicas > self.replicas:
            # TĂNG TÀI NGUYÊN (SCALE UP)
            mult_factor = {0: 1.0, 1: 1.5, 2: 2.0}[a_mult]
            diff = desired_replicas - self.replicas
            scale_up_amount = math.ceil(diff * mult_factor)
            
            # Chế độ tăng cường a_enh = 1 (Tấn công - Aggressive) hoặc khi CPU quá cao
            if a_enh == 1 or current_cpu > 0.85:
                scale_up_amount = max(scale_up_amount, 2)
                
            self.replicas = min(self.max_replicas, self.replicas + scale_up_amount)
            
            # Kích hoạt cooldown
            cooldown_steps = {0: 1, 1: 3, 2: 5}[a_lr]
            self.cooldown_remaining = cooldown_steps
            
        elif desired_replicas < self.replicas:
            # GIẢM TÀI NGUYÊN (SCALE DOWN)
            if self.cooldown_remaining == 0:
                # Chế độ tối ưu chi phí a_enh = 2 (Cost-saving) cho phép thu hồi nhanh
                if a_enh == 2:
                    scale_down_amount = self.replicas - desired_replicas
                else:
                    scale_down_amount = 1 # chế độ mặc định chỉ thu hồi từ từ từng replica để tránh dao động
                    
                self.replicas = max(self.min_replicas, self.replicas - scale_down_amount)
                
                # Kích hoạt cooldown
                cooldown_steps = {0: 1, 1: 3, 2: 5}[a_lr]
                self.cooldown_remaining = cooldown_steps
                
        # 6. Cập nhật trạng thái mới sau khi scale
        new_s_t = self._build_state_vector(workload_val)
        self.state_buffer.append(new_s_t)
        
        # 7. Tính toán Reward đa mục tiêu
        new_cpu = new_s_t[1]
        
        # Phạt SLA khi CPU quá tải (>90% làm tăng đáng kể tỉ lệ drop request/response time)
        sla_penalty = 0.0
        if new_cpu > 0.90:
            sla_penalty = 12.0 * (new_cpu - 0.90)
            
        # Phạt chi phí sử dụng tài nguyên (khuyến khích giữ ít replica nhất có thể)
        cost_penalty = 0.15 * self.replicas
        
        # Phạt dao động hạ tầng (phạt mỗi khi thay đổi số lượng replica)
        action_penalty = 0.0
        if self.replicas != self.prev_replicas:
            action_penalty = 0.3
            
        # Thưởng khi duy trì CPU trong vùng tối ưu hiệu năng và năng lượng (45% -> 75%)
        if 0.45 <= new_cpu <= 0.75:
            util_reward = 1.0
        else:
            dist = min(abs(new_cpu - 0.45), abs(new_cpu - 0.75))
            util_reward = math.exp(-dist / 0.25)
            
        reward = util_reward - sla_penalty - cost_penalty - action_penalty
        
        return self._get_tensor_state(), reward, self.current_step >= self.max_steps
