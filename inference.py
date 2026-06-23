import torch
import numpy as np
from models import PPOAgent
from collections import deque

class RealTimeAutoscaler:
    def __init__(self, model_path="ppo_double_lstm_scaler.pth"):
        # 1. Khởi tạo lại cấu trúc bộ não mạng Neural đã cải tiến
        self.agent = PPOAgent(input_dim=14, hidden_dim=128)
        
        # 2. Nạp trọng số đã được tối ưu hóa vào
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.agent.eval()  # Chuyển sang chế độ suy luận (Inference Mode)
        
        # 3. Cửa sổ trượt w=3 ghi nhớ lịch sử metrics
        self.state_buffer = deque(maxlen=3)
        for _ in range(3):
            self.state_buffer.append(np.zeros(14))

    def predict_next_action(self, live_metrics_14d):
        """
        Hàm nhận vào vector metrics 14 chiều lấy trực tiếp theo thời gian thực
        và trả về các quyết định cấu hình điều phối tài nguyên chi tiết.
        """
        # Đẩy trượt thông số mới vào bộ đệm
        self.state_buffer.append(live_metrics_14d)
        
        # Khôi phục cấu trúc ma trận Tensor chuẩn (1, 3, 14)
        state_tensor = torch.FloatTensor(np.array(self.state_buffer)).unsqueeze(0)
        
        # Thực hiện suy luận không tính toán đạo hàm
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action(state_tensor)
            
        # Giải mã 4 hành động cấu hình hạ tầng
        a_targ, a_lr, a_mult, a_enh = action[0].tolist()
        
        # Map sang giá trị cấu hình vật lý thực tế
        target_cpu = {0: 30, 1: 50, 2: 70, 3: 90}[a_targ]
        cooldown_steps = {0: 1, 1: 3, 2: 5}[a_lr]
        multiplier = {0: 1.0, 1: 1.5, 2: 2.0}[a_mult]
        enh_mode = {0: "Normal", 1: "Aggressive Scale-up", 2: "Cost-Saving Scale-down"}[a_enh]
        
        return {
            "target_cpu": target_cpu,
            "cooldown_steps": cooldown_steps,
            "multiplier": multiplier,
            "enh_mode": enh_mode
        }

# =========================================================
# VÒNG LẶP ĐIỀU KHIỂN HẠ TẦNG THỰC TẾ (CONTROL LOOP AGENT)
# =========================================================
if __name__ == "__main__":
    import os
    
    model_file = "ppo_double_lstm_scaler.pth"
    if not os.path.exists(model_file):
        print(f"[Error] Model weight file '{model_file}' not found. Please run 'python train.py' first to train the model!")
        exit(1)
        
    # Khởi tạo bộ điều phối cấu hình
    scaler = RealTimeAutoscaler(model_path=model_file)
    
    print("=== AI AUTOSCALER READY FOR PRODUCTION ===")
    
    # Giả lập vòng lặp quét thông số định kỳ hệ thống
    for step in range(1, 6):
        # 1. Đo đạc thông số thực tế từ máy chủ (đầu vào 14 chiều giả lập theo định dạng giàu trạng thái)
        # s_t[0]: workload_demand, s_t[1]: cpu_util, s_t[2]: replica_ratio, s_t[3]: target_ratio, ...
        live_sampled_metrics = np.zeros(14)
        live_sampled_metrics[0] = np.random.uniform(0.2, 0.8) # Workload demand
        live_sampled_metrics[1] = np.random.uniform(0.3, 0.9) # CPU util
        live_sampled_metrics[2] = 0.3                         # Replica ratio (3/10)
        live_sampled_metrics[3] = 0.5                         # Target ratio (50%)
        live_sampled_metrics[7] = 0.6                         # RAM
        live_sampled_metrics[8] = 0.15                        # Response Time
        
        # 2. Đưa vào cho mô hình AI dự đoán hành động cấu hình tối ưu
        decisions = scaler.predict_next_action(live_sampled_metrics)
        
        # 3. In ra quyết định điều phối hạ tầng
        print(f"\n[Step {step}] Received metrics from Prometheus...")
        print(f" -> Workload: {live_sampled_metrics[0]*100:.1f}% | Actual CPU: {live_sampled_metrics[1]*100:.1f}%")
        print(f" -> AI Decision:")
        print(f"    - Target CPU utilization: {decisions['target_cpu']}%")
        print(f"    - Cooldown: {decisions['cooldown_steps']} steps")
        print(f"    - Scale multiplier: {decisions['multiplier']}x")
        print(f"    - Operating mode: {decisions['enh_mode']}")
        
        # Trong thực tế, bạn sẽ chèn lệnh gọi API hạ tầng ở đây:
        # e.g. client.patch_namespaced_horizontal_pod_autoscaler(...)
