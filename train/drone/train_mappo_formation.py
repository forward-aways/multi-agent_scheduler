"""
优化的多智能体无人机队形保持任务训练脚本 - MAPPO算法（离散动作版）
与包围任务保持一致的动作空间设计
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

from environments.multi_agent_drone_env import MultiAgentDroneEnv
from utils.logging_config import training_logger


class ActorNetwork(nn.Module):
    """Actor 网络 - 策略网络（MAPPO）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)  # 输出离散动作的概率分布
        )
    
    def forward(self, state):
        return self.network(state)
    
    def get_action_and_log_prob(self, state):
        """获取动作和对数概率（离散动作）"""
        logits = self.network(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络（MAPPO）"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class DroneMAPPOAgent:
    """无人机MAPPO智能体（离散动作版）"""
    
    def __init__(self, agent_id: int, state_dim: int, action_dim: int, config: dict):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建网络
        self.actor = ActorNetwork(state_dim, action_dim, config.get('hidden_dim', 256))
        self.critic = CriticNetwork(state_dim, config.get('hidden_dim', 256))
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get('actor_lr', 3e-4))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.get('critic_lr', 1e-3))
        
        # MAPPO超参数
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.05)  # 增加熵系数，鼓励探索
        self.value_coef = config.get('value_coef', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        
        # 经验回放缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, training: bool = True):
        """选择动作（离散动作）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
                value = self.critic(state_tensor)
                return action.item(), log_prob.item(), value.item()
            else:
                logits = self.actor(state_tensor)
                action = torch.argmax(logits, dim=-1)
                return action.item()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """存储转换"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """更新网络（MAPPO算法 - 离散动作）"""
        if len(self.states) < self.mini_batch_size:
            return
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        dones = torch.FloatTensor(np.array(self.dones))
        
        # 计算优势函数（GAE）
        advantages = self._compute_gae(rewards, values.numpy(), dones.numpy())
        returns = torch.from_numpy(advantages) + values
        
        # 归一化优势函数
        advantages = (torch.from_numpy(advantages) - torch.from_numpy(advantages).mean()) / (torch.from_numpy(advantages).std() + 1e-8)
        
        # PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(states))
            
            # 小批量更新
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的动作概率（离散动作）
                logits = self.actor(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                new_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(new_values, batch_returns)
                
                # 更新Actor（只包含actor_loss和entropy）
                actor_total_loss = actor_loss - self.entropy_coef * entropy
                self.actor_optimizer.zero_grad()
                actor_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # 更新Critic（只包含critic_loss）
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def _compute_gae(self, rewards, values, dones):
        """计算广义优势估计（GAE）"""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


class MAPPOTrainer:
    """MAPPO训练器（离散动作版）"""
    
    def __init__(self, env: MultiAgentDroneEnv, config: dict, load_existing: bool = False):
        self.env = env
        self.config = config
        self.num_drones = env.num_drones
        
        # 创建智能体
        self.agents = {}
        # 获取环境的离散动作数量
        action_dim = env.num_actions  # 27个离散动作
        
        for i in range(self.num_drones):
            # 计算状态维度
            obs, _ = env.reset()
            state_dim = obs[f'drone_{i}'].shape[0]
            
            agent = DroneMAPPOAgent(i, state_dim, action_dim, config)
            self.agents[f'drone_{i}'] = agent
        
        # 如果存在已有模型，加载它
        if load_existing:
            self.load_existing_models()
        
        self.episode_rewards = []
        self.best_reward = -float('inf')
        self.best_episode = 0
    
    def load_existing_models(self):
        """加载已存在的模型"""
        model_dir = Path(__file__).parent.parent.parent / "models" / "multi_agent_drone" / "mappo" / "formation" / self.env.formation_type
        
        if not model_dir.exists():
            print(f"模型目录不存在: {model_dir}")
            return
        
        print(f"加载已有模型: {model_dir}")
        loaded_count = 0
        for i in range(self.num_drones):
            model_path = model_dir / f"drone_{i}_agent.pth"
            if model_path.exists():
                try:
                    self.agents[f'drone_{i}'].load(str(model_path))
                    loaded_count += 1
                except Exception as e:
                    print(f"加载模型 {model_path} 失败: {e}")
        
        if loaded_count > 0:
            print(f"✓ 成功加载 {loaded_count} 个模型，继续训练...")
        else:
            print("没有找到已有模型，从头开始训练...")
    
    def train(self, episodes: int = 500):
        """训练"""
        print(f"开始训练，总回合数: {episodes}")
        print(f"{'='*60}")
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            step_count = 0
            
            while step_count < self.env.max_steps:
                # 每个智能体选择动作
                actions = {}
                for i in range(self.num_drones):
                    drone_id = f'drone_{i}'
                    state = obs[drone_id]
                    action, log_prob, value = self.agents[drone_id].select_action(state, training=True)
                    actions[drone_id] = action
                    
                    # 存储转换
                    self.agents[drone_id].store_transition(state, action, log_prob, 0, value, False)
                
                # 执行动作
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # 更新奖励
                for i in range(self.num_drones):
                    drone_id = f'drone_{i}'
                    self.agents[drone_id].rewards[-1] = rewards[drone_id]
                    self.agents[drone_id].dones[-1] = terminated[drone_id] or truncated[drone_id]
                
                total_reward += sum(rewards.values())
                obs = next_obs
                step_count += 1
                
                if all(terminated.values()) or all(truncated.values()):
                    break
            
            # 更新网络
            for i in range(self.num_drones):
                self.agents[f'drone_{i}'].update()
            
            self.episode_rewards.append(total_reward)
            
            # 保存最佳模型
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_episode = episode
                self.save_best_models()
            
            # 打印进度
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(f"回合 {episode + 1}/{episodes}, 平均奖励: {avg_reward:.2f}, 最佳奖励: {self.best_reward:.2f} (回合 {self.best_episode})")
        
        print(f"{'='*60}")
        print(f"训练完成！最佳奖励: {self.best_reward:.2f} (回合 {self.best_episode})")
    
    def save_best_models(self):
        """保存最佳模型"""
        model_dir = Path(__file__).parent.parent.parent / "models" / "multi_agent_drone" / "mappo" / "formation" / self.env.formation_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.num_drones):
            model_path = model_dir / f"drone_{i}_agent.pth"
            self.agents[f'drone_{i}'].save(str(model_path))


def main():
    parser = argparse.ArgumentParser(description='训练无人机队形保持任务')
    parser.add_argument('--formation', type=str, default='triangle', choices=['triangle', 'v_shape', 'line', 'circle'],
                       help='队形类型')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    args = parser.parse_args()
    
    # 环境配置
    env_config = {
        'num_drones': 3,
        'max_speed': 2.0,  # 与测试时保持一致
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'formation',
        'formation_type': args.formation,
        'max_steps': 200
    }
    
    # 创建环境
    env = MultiAgentDroneEnv(env_config)
    
    # 配置参数（优化版）
    config = {
        'model_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "models", "multi_agent_drone", "mappo", "formation", args.formation),
        'actor_lr': 1e-4,  # 降低学习率，更稳定
        'critic_lr': 5e-4,  # 降低学习率
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.15,  # 减小裁剪范围，更保守
        'entropy_coef': 0.08,  # 增加熵系数，鼓励更多探索
        'value_coef': 0.5,
        'ppo_epochs': 15,  # 增加PPO更新次数
        'mini_batch_size': 32,  # 减小batch size，更精细的更新
    }
    
    # 创建训练器（从头训练，不加载已有模型）
    trainer = MAPPOTrainer(env, config, load_existing=False)
    
    # 训练
    trainer.train(episodes=args.episodes)
    
    print(f"\n训练完成！模型已保存到: {config['model_dir']}")


if __name__ == "__main__":
    main()
