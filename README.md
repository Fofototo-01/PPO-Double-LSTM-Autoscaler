# KVM/Kubernetes Autoscaler using PPO & Double-Stacked LSTM
Dự án sử dụng Deep Reinforcement Learning để tự động tối ưu hóa tài nguyên CPU cho cụm máy chủ điện toán đám mây.

## 🧠 Kiến trúc Mô hình
* **Thuật toán:** Proximal Policy Optimization (PPO) kết hợp Soft-Attention.
* **Trích xuất đặc trưng:** Mạng Neural Double-Stacked LSTM giải quyết bài toán Temporal Blindness.
* **Không gian hành động:** Multi-Discrete (Điều chỉnh HPA Target, LR, Multiplier).

## 📊 Kết quả Huấn luyện
Mô hình hội tụ xuất sắc sau 1000 Episodes trên tập dữ liệu thực tế:
*(Bạn kéo thả 2 file ảnh .png biểu đồ Reward và Loss của bạn vào thẳng trang GitHub Web, nó sẽ tự sinh ra link ảnh ở đây)*

## 🚀 Cách sử dụng
1. Clone repo này về máy.
2. Cài đặt thư viện: `pip install torch pandas numpy matplotlib`
3. Chạy quá trình huấn luyện: `python train.py`
4. Để test trên file nhúng: `python inference.py`