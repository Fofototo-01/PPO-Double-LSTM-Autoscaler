import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
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
print(f"Training device: {device}")

env = KubernetesEnv(csv_path="/content/drive/MyDrive/Dataset/data.csv", max_steps=ROLLOUT_STEPS)
agent = PPOAgent(input_dim=14, hidden_dim=128).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPISODES, eta_min=1e-5)

def train():
    # Khởi tạo các mảng lưu lịch sử để phục vụ vẽ đồ thị báo cáo
    reward_history = []
    loss_history = []
    actor_loss_history = []
    critic_loss_history = []
    entropy_history = []
    
    print("=== STARTING TRAINING PROCESS WITH IMPROVED CONFIGURATION ===")
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset().to(device)
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        # PHA 1: THU THẬP TRẢI NGHIỆM (ROLLOUT)
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
        # Chuẩn hóa Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PHA 3: TỐI ƯU HÓA TRỌNG SỐ PPO BẰNG MINI-BATCH
        epoch_losses = []
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropies = []
        
        dataset_size = len(rewards)
        indices = np.arange(dataset_size)
        
        for _ in range(UPDATE_EPOCHS):
            # Trộn dữ liệu ngẫu nhiên trước mỗi Epoch cập nhật
            np.random.shuffle(indices)
            
            # Chia nhỏ Rollout thành các mini-batch
            for start_idx in range(0, dataset_size, BATCH_SIZE):
                batch_indices = indices[start_idx:start_idx + BATCH_SIZE]
                if len(batch_indices) < 32:  # Bỏ qua các batch quá nhỏ ở cuối để tránh nhiễu nhiễu đạo hàm
                    continue
                    
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_log_probs = log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]
                
                # Tính giá trị dự báo mới của chính sách hiện tại
                _, new_log_probs, entropies, new_values = agent.get_action(mb_states, mb_actions)
                new_values = new_values.squeeze()
                
                # Tỷ lệ chính sách mới / cũ
                ratio = torch.exp(new_log_probs - mb_log_probs)
                
                # Cắt (clip) hàm mục tiêu Actor
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Hàm Critic Loss (MSE)
                critic_loss = F.mse_loss(new_values, mb_returns)
                
                # Entropy khuyến khích khám phá hành động mới
                total_entropy = torch.stack(entropies).sum(dim=0).mean()
                
                # Hàm loss tổng hợp của PPO
                loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * total_entropy
                
                optimizer.zero_grad()
                loss.backward()
                # Cắt bớt Gradient (Clipping) bảo vệ mạng LSTM khỏi bùng nổ đạo hàm
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                epoch_entropies.append(total_entropy.item())
                
        # Cập nhật tốc độ học theo scheduler
        scheduler.step()
        
        # Lưu lại lịch sử
        reward_history.append(sum(rewards))
        loss_history.append(np.mean(epoch_losses))
        actor_loss_history.append(np.mean(epoch_actor_losses))
        critic_loss_history.append(np.mean(epoch_critic_losses))
        entropy_history.append(np.mean(epoch_entropies))
        
        if episode % 10 == 0 or episode == 1:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"Episode {episode:04d}/{NUM_EPISODES} | "
                  f"Reward: {sum(rewards):.2f} | "
                  f"Loss: {np.mean(epoch_losses):.4f} | "
                  f"Actor Loss: {np.mean(epoch_actor_losses):.4f} | "
                  f"Critic Loss: {np.mean(epoch_critic_losses):.4f} | "
                  f"LR: {curr_lr:.6f}")

    # ==========================================
    # PHẦN XUẤT FILE ĐỒ THỊ VÀ MÔ HÌNH SAU KHI HOÀN THÀNH
    # ==========================================
    print("\n=== TRAINING COMPLETED! EXPORTING ARTIFACTS ===")
    
    # 1. Lưu cấu trúc trọng số mạng neural đã cải tiến
    torch.save(agent.state_dict(), "ppo_double_lstm_scaler.pth")
    print("-> Saved model weight: ppo_double_lstm_scaler.pth")
    
    # 2. Vẽ và xuất biểu đồ xung lực Reward
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, color='#1f77b4', linewidth=1.5, label='Cumulative Reward')
    # Thêm đường trung bình trượt 50 chặng để thấy rõ xu hướng hội tụ
    if len(reward_history) >= 50:
        smoothed_reward = np.convolve(reward_history, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(reward_history)), smoothed_reward, color='#ff7f0e', linewidth=2, label='Moving Avg (50)')
    plt.title('Convergence Curve: Cumulative Reward Over Episodes', fontsize=12, fontweight='bold')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Total Reward', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('ppo_reward_convergence.pdf', bbox_inches='tight')
    plt.savefig('ppo_reward_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved reward plot: ppo_reward_convergence.pdf/.png")

    # 3. Vẽ và xuất biểu đồ hội tụ hàm Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#d62728', linewidth=1.2, alpha=0.4, label='Total Loss (Raw)')
    # Thêm đường trung bình trượt 10 chặng cho Loss
    if len(loss_history) >= 10:
        smoothed_loss = np.convolve(loss_history, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(loss_history)), smoothed_loss, color='#d62728', linewidth=2, label='Total Loss (Smoothed)')
    plt.title('Model Optimization: Total PPO Loss Over Episodes', fontsize=12, fontweight='bold')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Loss Value', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('ppo_loss_convergence.pdf', bbox_inches='tight')
    plt.savefig('ppo_loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved loss plot: ppo_loss_convergence.pdf/.png")

    # 4. Vẽ biểu đồ các thành phần Loss chi tiết (Mới bổ sung để báo cáo trực quan)
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(actor_loss_history, color='purple', label='Actor Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.title('Detailed Loss Components Analysis')
    
    plt.subplot(3, 1, 2)
    plt.plot(critic_loss_history, color='orange', label='Critic Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(entropy_history, color='green', label='Entropy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel('Episode')
    
    plt.tight_layout()
    plt.savefig('ppo_detailed_losses.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved detailed loss components plot: ppo_detailed_losses.png")

if __name__ == "__main__":
    train()
