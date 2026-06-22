import torch
import numpy as np
from models import PPOAgent
from collections import deque

class RealTimeAutoscaler:
    def __init__(self, model_path="ppo_double_lstm_scaler.pth"):
        # 1. Khởi tạo lại đúng cấu trúc bộ não mạng Neural cũ
        self.agent = PPOAgent(input_dim=14, hidden_dim=128)
        
        # 2. Nạp toàn bộ trọng số đã được tối ưu hóa từ 1000 vòng train vào
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.agent.eval()  # Chuyển sang chế độ suy luận (Inference Mode), tắt dropout.
        
        # 3. Duy trì cửa sổ trượt w=3 để ghi nhớ lịch sử metrics
        self.state_buffer = deque(maxlen=3)
        for _ in range(3):
            self.state_buffer.append(np.zeros(14))

    def predict_next_action(self, live_metrics_14d):
        """
        Hàm nhận vào vector metrics 14 chiều lấy trực tiếp theo thời gian thực 
        (Ví dụ trích xuất từ Prometheus API hoặc KVM Agent gộp về)
        """
        # Đẩy trượt thông số mới vào bộ đệm
        self.state_buffer.append(live_metrics_14d)
        
        # Khôi phục cấu trúc ma trận Tensor chuẩn (1, 3, 14)
        state_tensor = torch.FloatTensor(np.array(self.state_buffer)).unsqueeze(0)
        
        # Thực hiện suy luận không tính toán đạo hàm (Tiết kiệm tài nguyên tối đa)
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action(state_tensor)
            
        # Giải mã hành động cấu hình hạ tầng
        a_targ, a_lr, a_mult, a_enh = action[0].tolist()
        
        target_mapping = {0: 30, 1: 50, 2: 70, 3: 90}
        recommended_cpu_target = target_mapping[a_targ]
        
        return recommended_cpu_target

# =========================================================
# VÒNG LẶP ĐIỀU KHIỂN HẠ TẦNG THỰC TẾ (CONTROL LOOP AGENT)
# =========================================================
if __name__ == "__main__":
    # Khởi tạo bộ điều phối cấu hình
    scaler = RealTimeAutoscaler(model_path="ppo_double_lstm_scaler.pth")
    
    print("=== BỘ ĐIỀU PHỐI AI ĐÃ SẴN SÀNG CHẠY TRÊN HỆ THỐNG PRODUCTION ===")
    
    # Giả lập vòng lặp quét thông số định kỳ hệ thống
    # Trong thực tế, đây sẽ là hàm while True: gọi API sau mỗi 5 hoặc 15 phút
    for step in range(1, 5):
        # 1. Đo đạc thông số thực tế từ máy chủ tại thời điểm hiện tại
        # Giả lập vector thu thập từ hệ thống giám sát
        live_sampled_metrics = np.random.uniform(0.1, 0.8, 14) 
        
        # 2. Đưa vào cho mô hình AI dự đoán hành động cấu hình tối ưu
        action_decision = scaler.predict_next_action(live_sampled_metrics)
        
        # 3. Thực thi hành động xuống hạ tầng cụm node thông qua CLI/API
        print(f"[Nhịp {step}] Nhận thông số hạ tầng -> AI ra quyết định điều chỉnh Target CPU: {action_decision}%")
        # Chỗ này bạn sẽ chèn lệnh gọi API thực tế: os.system(f"openstack server set ...") hoặc k8s API
