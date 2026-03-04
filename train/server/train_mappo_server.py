"""
多智能体服务器调度训练脚本 - MAPPO算法
使用MAPPO算法训练多智能体协作系统
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from environments.multi_agent_server_env import MultiAgentServerEnv
from utils.logging_config import training_logger


class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络（MAPPO）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def get_action_and_log_prob(self, state):
        """获取动作和对数概率"""
        logits = self.network(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
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


class MAPPOAgent:
    """MAPPO单个智能体"""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: dict):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_old = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # 初始化actor_old为actor的副本
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get('actor_lr', 3e-4))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.get('critic_lr', 1e-3))
        
        # PPO超参数
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        
        # 经验存储
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def select_action(self, state: torch.Tensor, training: bool = True):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(state)
                value = self.critic(state)
                return action.item(), log_prob.item(), value.item()
            else:
                logits = self.actor(state)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1)
                return action.item(), 0.0, 0.0
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """存储转换"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """计算广义优势估计（GAE）"""
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns
    
    def update(self, next_value):
        """更新网络"""
        # 计算GAE和回报
        advantages, returns = self.compute_gae(next_value)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.ppo_epochs):
            # 创建mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的log_prob和熵
                logits = self.actor(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算PPO损失（裁剪）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # 计算Critic损失
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # 更新旧策略
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # 清空缓冲区
        self.clear_buffer()
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_old.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class MAPPOTrainer:
    """MAPPO训练器"""
    
    def __init__(self, env: MultiAgentServerEnv, config: dict):
        self.env = env
        self.config = config
        self.num_agents = env.num_servers  # 添加智能体数量属性
        
        # 创建智能体
        self.agents = []
        for i in range(env.num_servers):
            agent_id = f"server_{i}"
            # 获取观测空间和动作空间
            state_dim = env.observation_spaces[f'server_{i}'].shape[0]
            action_dim = env.action_spaces[f'server_{i}'].n
            agent = MAPPOAgent(agent_id, state_dim, action_dim, config)
            self.agents.append(agent)
    
    def train(self, episodes: int):
        """训练智能体"""
        training_logger.info(f"开始训练多智能体系统（MAPPO），共 {episodes} 回合")
        
        all_rewards = []
        
        for episode in range(episodes):
            # 重置环境
            states, _ = self.env.reset()
            episode_rewards = [0.0] * self.env.num_servers
            episode_dones = [False] * self.env.num_servers
            
            # 收集一个episode的经验
            for step in range(self.env.max_steps):
                # 选择动作
                actions = {}
                for i, agent in enumerate(self.agents):
                    state = states[f'server_{i}']
                    action, log_prob, value = agent.select_action(state, training=True)
                    agent.store_transition(state, action, log_prob, 0.0, value, False)
                    actions[f'server_{i}'] = action
                
                # 执行动作
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # 更新奖励
                for i, reward in enumerate(rewards.values()):
                    episode_rewards[i] += reward
                    self.agents[i].rewards[-1] = reward
                
                # 检查是否结束
                all_done = all(terminated.values()) or all(truncated.values()) or step == self.env.max_steps - 1
                
                if all_done:
                    # 计算最终价值
                    next_values = []
                    for i, agent in enumerate(self.agents):
                        with torch.no_grad():
                            next_state = torch.FloatTensor(next_states[f'server_{i}']).unsqueeze(0)
                            next_value = agent.critic(next_state).item()
                            next_values.append(next_value)
                    
                    # 更新所有智能体
                    for i, agent in enumerate(self.agents):
                        agent.update(next_values[i])
                    break
                
                states = next_states
            
            # 计算平均奖励
            avg_reward = np.mean(episode_rewards)
            all_rewards.append(avg_reward)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                recent_rewards = all_rewards[-10:]
                training_logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {np.mean(recent_rewards):.2f}")
        
        # 保存模型
        for i, agent in enumerate(self.agents):
            model_path = os.path.join(self.config.get('model_dir', 'models/multi_agent_server/mappo'), f"server_{i}_agent.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            agent.save_model(model_path)
            training_logger.info(f"保存模型: {model_path}")
        
        training_logger.info("训练完成!")
        training_logger.info(f"平均奖励: {np.mean(all_rewards):.2f}")
        training_logger.info(f"最终奖励: {all_rewards[-1]:.2f}")
        
        return all_rewards


def main():
    """主函数"""
    # 环境配置
    env_config = {
        'num_servers': 5,
        'server_cpu_capacity': 100.0,
        'server_memory_capacity': 100.0,
        'server_max_tasks': 10,
        'task_generation_rate': 5,
        'max_pending_tasks': 15,
        'max_steps': 100
    }
    
    # 创建环境
    env = MultiAgentServerEnv(env_config)
    
    # 训练配置
    config = {
        'model_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "multi_agent_server", "mappo"),
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'ppo_epochs': 10,
        'mini_batch_size': 64
    }
    
    # 创建训练器
    trainer = MAPPOTrainer(env, config)
    
    # 训练
    training_episodes = 600
    rewards = trainer.train(episodes=training_episodes)


if __name__ == "__main__":
    main()