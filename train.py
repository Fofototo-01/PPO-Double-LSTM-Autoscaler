import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # Thêm thư viện vẽ đồ thị chuyên nghiệp
import torch.nn.functional as F
from models import PPOAgent
from environment import KubernetesEnv

# HYPERPARAMETERS
LEARNING_RATE = 2e-4
GAMMA = 0.99
GAE_LAMBDA = 0.93
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.01
UPDATE_EPOCHS = 10
BATCH_SIZE = 128
ROLLOUT_STEPS = 512
NUM_EPISODES = 1000  # Cấu hình 1000 vòng lặp cho giai đoạn train thật

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hệ thống đang thực thi huấn luyện trên thiết bị: {device}")

env = KubernetesEnv(csv_path="/content/drive/MyDrive/Dataset/data.csv", max_steps=ROLLOUT_STEPS)
agent = PPOAgent(input_dim=14, hidden_dim=128).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

def train():
    # Khởi tạo các mảng lưu lịch sử để phục vụ vẽ đồ thị báo cáo
    reward_history = []
    loss_history = []
    
    print("=== KHỞI CHẠY TIẾN TRÌNH HUẤN LUYỆN 1000 EPISODES ===")
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset().to(device)
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        # PHA 1: THU THẬP TRẢI NGHIỆM
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action(state)
                
            next_state, reward, done = env.step(action)
            next_state = next_state.to(device)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            state = next_state
            if done:
                break

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        values = torch.cat(values, dim=0).squeeze()
        
        # PHA 2: TÍNH TOÁN LỢI THẾ GAE
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0
        next_value = 0 if dones[-1] else values[-1].item()
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t].item()
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t].item()
            next_value = values[t].item()
            
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PHA 3: TỐI ƯU HÓA TRỌNG SỐ PPO
        epoch_losses = []
        for _ in range(UPDATE_EPOCHS):
            _, new_log_probs, entropies, new_values = agent.get_action(states, actions)
            new_values = new_values.squeeze()
            
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(new_values, returns)
            total_entropy = torch.stack(entropies).sum(dim=0).mean()
            loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * total_entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        # Lưu lại giá trị trung bình của chu kỳ này
        reward_history.append(sum(rewards))
        loss_history.append(np.mean(epoch_losses))
        
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode:04d}/{NUM_EPISODES} | Tích lũy Reward: {sum(rewards):.2f} | Tổng Loss: {np.mean(epoch_losses):.4f}")

    # ==========================================
    # PHẦN XUẤT FILE SAU KHI HOÀN THÀNH 1000 VÒNG
    # ==========================================
    print("\n=== HOÀN THÀNH HUẤN LUYỆN! ĐANG XUẤT TÀI NGUYÊN ===")
    
    # 1. Lưu cấu trúc trọng số mạng neural (Bộ não AI)
    torch.save(agent.state_dict(), "ppo_double_lstm_scaler.pth")
    print("-> Đã lưu file mô hình: ppo_double_lstm_scaler.pth")
    
    # 2. Vẽ và xuất biểu đồ xung lực Reward
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, color='#1f77b4', linewidth=1.5, label='Cumulative Reward')
    plt.title('Convergence Curve: Cumulative Reward Over Episodes', fontsize=12, fontweight='bold')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Total Reward', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('ppo_reward_convergence.pdf', bbox_inches='tight')  # Xuất đuôi .pdf chất lượng cao cho báo cáo
    plt.savefig('ppo_reward_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Đã xuất đồ thị Reward: ppo_reward_convergence.pdf/.png")

    # 3. Vẽ và xuất biểu đồ hội tụ hàm Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#d62728', linewidth=1.5, label='Training Loss')
    plt.title('Model Optimization: Total PPO Loss Over Episodes', fontsize=12, fontweight='bold')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Loss Value', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('ppo_loss_convergence.pdf', bbox_inches='tight')
    plt.savefig('ppo_loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Đã xuất đồ thị Loss: ppo_loss_convergence.pdf/.png")

if __name__ == "__main__":
    train()
