"""
优化的多智能体无人机协同包围任务训练脚本 - MAPPO算法
解决训练和测试配置不一致的问题
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
    """无人机MAPPO智能体"""
    
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
        self.entropy_coef = config.get('entropy_coef', 0.01)
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
                
                # 计算新的动作概率
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
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # 更新Critic
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
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


class MAPPOTrainer:
    """MAPPO训练器"""
    
    def __init__(self, env: MultiAgentDroneEnv, config: dict):
        self.env = env
        self.config = config
        self.num_drones = env.num_drones
        
        # 创建智能体
        self.agents = []
        # 获取环境的离散动作数量
        action_dim = env.num_actions  # 27个离散动作
        
        for i in range(self.num_drones):
            # 计算状态维度
            obs, _ = env.reset()
            state_dim = obs[f'drone_{i}'].shape[0]
            
            agent = DroneMAPPOAgent(i, state_dim, action_dim, config)
            self.agents.append(agent)
        
        self.episode_rewards = []
        self.best_reward = -float('inf')  # 最佳奖励
        self.best_episode = 0  # 最佳回合
    
    def train(self, episodes: int = 500):
        """训练"""
        print(f"开始训练，总回合数: {episodes}")
        print(f"{'='*60}")
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            
            for step in range(self.env.max_steps):
                # 收集动作
                actions = {}
                log_probs = []
                values = []
                
                for i, agent in enumerate(self.agents):
                    agent_id = f'drone_{i}'
                    # 离散动作：select_action 返回动作索引
                    action_idx, log_prob, value = agent.select_action(obs[agent_id], training=True)
                    actions[agent_id] = action_idx
                    log_probs.append(log_prob)
                    values.append(value)
                
                # 执行动作
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # 计算总奖励
                step_reward = sum(rewards.values())
                episode_reward += step_reward
                
                # 存储转换
                for i, agent in enumerate(self.agents):
                    agent_id = f'drone_{i}'
                    agent.store_transition(
                        obs[agent_id],
                        actions[agent_id],  # 存储离散动作索引
                        log_probs[i],
                        rewards[agent_id],
                        values[i],
                        all(terminated.values()) or all(truncated.values())
                    )
                
                obs = next_obs
                
                if all(terminated.values()) or all(truncated.values()):
                    break
            
            # 更新网络
            for agent in self.agents:
                agent.update()
            
            # 记录奖励
            avg_episode_reward = episode_reward / self.num_drones
            self.episode_rewards.append(avg_episode_reward)
            
            # 检查是否是最佳模型
            if avg_episode_reward > self.best_reward:
                self.best_reward = avg_episode_reward
                self.best_episode = episode + 1
                # 保存最佳模型
                model_dir = self.config.get('model_dir', 'models/encirclement')
                os.makedirs(model_dir, exist_ok=True)
                for i, agent in enumerate(self.agents):
                    model_path = os.path.join(model_dir, f'drone_{i}_agent.pth')
                    agent.save(model_path)
                training_logger.info(f"保存最佳模型 (回合 {self.best_episode}, 奖励: {self.best_reward:.2f}): {model_dir}")
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                max_reward = np.max(self.episode_rewards)
                print(f"进度: [{episode+1}/{episodes}] ({(episode+1)/episodes*100:.1f}%) | 本回合奖励: {avg_episode_reward:.2f} | 最近100回合平均: {avg_reward:.2f} | 历史最高: {max_reward:.2f} | 最佳模型: 回合{self.best_episode}({self.best_reward:.2f})")
                training_logger.info(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Best: {self.best_reward:.2f}")
        
        # 打印训练结果
        avg_reward = np.mean(self.episode_rewards)
        final_reward = self.episode_rewards[-1]
        max_reward = np.max(self.episode_rewards)
        
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"最终奖励: {final_reward:.2f}")
        print(f"最高奖励: {max_reward:.2f}")
        print(f"最佳模型: 回合 {self.best_episode} (奖励: {self.best_reward:.2f})")
        print(f"模型已保存到: {self.config.get('model_dir', 'models/encirclement')}")
        print(f"{'='*60}\n")
        
        training_logger.info(f"协同包围任务训练完成!")
        training_logger.info(f"平均奖励: {avg_reward:.2f}")
        training_logger.info(f"最终奖励: {final_reward:.2f}")
        training_logger.info(f"最高奖励: {max_reward:.2f}")
        training_logger.info(f"最佳模型: 回合 {self.best_episode} (奖励: {self.best_reward:.2f})")
        
        return self.episode_rewards


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练无人机协同包围任务')
    parser.add_argument('--episodes', type=int, default=500,
                       help='训练回合数')
    args = parser.parse_args()
    
    # 环境配置（与测试时保持一致）
    env_config = {
        'num_drones': 3,
        'max_speed': 2.0,  # 与测试时保持一致
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'encirclement',
        'max_steps': 200
    }
    
    env = MultiAgentDroneEnv(env_config)
    
    config = {
        'model_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "models", "multi_agent_drone", "mappo", "encirclement"),
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.05,  # 增加熵系数，鼓励探索
        'value_coef': 0.5,
        'ppo_epochs': 10,
        'mini_batch_size': 64,
        # 离散动作不需要 action_std
    }
    
    trainer = MAPPOTrainer(env, config)
    
    training_episodes = args.episodes
    rewards = trainer.train(episodes=training_episodes)
    
    print(f"协同包围任务训练完成，模型已保存到 {config['model_dir']}")


if __name__ == "__main__":
    main()
