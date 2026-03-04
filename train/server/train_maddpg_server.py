"""
多智能体服务器调度训练脚本
使用MADDPG算法训练多智能体协作系统
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


# 定义经验回放缓冲区
Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones'])


class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络（优化版）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        # 优化：使用Tanh激活函数，梯度更稳定
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def get_action_probs(self, state):
        """获取动作概率分布"""
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
    def get_action_and_log_prob(self, state):
        """获取动作和对数概率"""
        probs = self.get_action_probs(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络（优化版）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        # 输入：联合状态 + 联合动作
        critic_input_dim = state_dim + action_dim
        
        # 优化：使用Tanh激活函数
        self.network = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # 输出Q值
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class MADDPGAgent:
    """MADDPG单个智能体（优化版）"""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: dict):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        
        # 优化器（优化学习率）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get('actor_lr', 3e-4))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.get('critic_lr', 1e-3))
        
        # 硬更新目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # 超参数（优化版）
        self.gamma = config.get('gamma', 0.99)  # 提高折扣因子
        self.tau = config.get('tau', 0.005)  # 降低软更新参数
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.entropy_coef = config.get('entropy_coef', 0.01)  # 添加熵系数
        
        # 梯度裁剪阈值
        self.grad_clip = config.get('grad_clip', 0.5)
        
        # 经验回放
        self.replay_buffer = deque(maxlen=config.get('buffer_size', 100000))
        self.batch_size = config.get('batch_size', 128)
        
    def hard_update(self, target, source):
        """硬更新：将source网络参数复制到target网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source):
        """软更新：按比例混合source和target网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> int:
        """选择动作（优化版）"""
        self.actor.eval()
        with torch.no_grad():
            # 优化：使用概率采样 + epsilon-greedy
            if add_noise and random.random() < self.epsilon:
                # 探索：从概率分布中采样（而不是完全随机）
                action, _ = self.actor.get_action_and_log_prob(state)
                action = action.item()
            else:
                # 利用：选择概率最大的动作
                probs = self.actor.get_action_probs(state)
                action = torch.argmax(probs).item()
        
        return action
    
    def add_experience(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """添加经验到回放缓冲"""
        # 将动作转换为one-hot编码
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1
        
        experience = (state, action_onehot, reward, next_state, done)
        self.replay_buffer.append(experience)
    
    def train_on_batch(self, batch_states: torch.Tensor, batch_actions: torch.Tensor, 
                       batch_rewards: torch.Tensor, batch_next_states: torch.Tensor, 
                       batch_dones: torch.Tensor, all_agents=None):
        """使用批量数据训练智能体（优化版）"""
        
        # 训练Critic
        # 计算目标Q值
        with torch.no_grad():
            # 获取所有智能体的下一个动作
            next_action_probs_all = []
            for i, agent in enumerate(all_agents):
                next_action_probs = agent.actor_target(batch_next_states)
                next_action = torch.argmax(next_action_probs, dim=1, keepdim=True)
                # 转换为one-hot
                next_action_onehot = torch.zeros_like(next_action_probs)
                next_action_onehot.scatter_(1, next_action, 1)
                next_action_probs_all.append(next_action_onehot)
            
            next_actions_all = torch.stack(next_action_probs_all, dim=1)
            
            # 计算目标Q值
            target_q = self.critic_target(batch_next_states.view(-1, self.state_dim), 
                                         next_actions_all[:, int(self.agent_id.split('_')[1]), :].view(-1, self.action_dim))
            target_q = batch_rewards + (self.gamma * target_q * ~batch_dones)
        
        # 当前Q值
        current_q = self.critic(batch_states.view(-1, self.state_dim), 
                               batch_actions.view(-1, self.action_dim))
        
        # Critic损失
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        
        # 更新Critic（添加梯度裁剪）
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # 训练Actor（添加熵正则化）
        current_action_probs = self.actor.get_action_probs(batch_states)
        current_action = torch.argmax(current_action_probs, dim=1, keepdim=True)
        current_action_onehot = torch.zeros_like(current_action_probs)
        current_action_onehot.scatter_(1, current_action, 1)
        
        # 计算熵
        entropy = -(current_action_probs * torch.log(current_action_probs + 1e-8)).sum(dim=-1).mean()
        
        # Actor损失（添加熵正则化）
        actor_loss = -self.critic(batch_states.view(-1, self.state_dim), 
                                 current_action_onehot.view(-1, self.action_dim)).mean() - self.entropy_coef * entropy
        
        # 更新Actor（添加梯度裁剪）
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class MADDPGTrainer:
    """MADDPG训练器"""
    
    def __init__(self, env: MultiAgentServerEnv, config: dict):
        self.env = env
        self.config = config
        
        # 获取环境信息
        dummy_obs, _ = self.env.reset()
        self.num_agents = len(dummy_obs)
        self.state_dims = [dummy_obs[f'server_{i}'].shape[0] for i in range(self.num_agents)]
        self.action_dims = [3] * self.num_agents  # 每个智能体有3个动作
        
        # 创建智能体
        self.agents = []
        for i in range(self.num_agents):
            agent_config = config.copy()
            agent = MADDPGAgent(
                agent_id=f"server_{i}",
                state_dim=self.state_dims[i],
                action_dim=self.action_dims[i],
                config=agent_config
            )
            self.agents.append(agent)
    
    def train(self, episodes: int):
        """训练智能体"""
        training_logger.info(f"开始训练多智能体系统（优化版MADDPG），共 {episodes} 回合")
        
        episode_rewards = []
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            step_count = 0
            
            # 转换观测为tensor
            states = {f'server_{i}': torch.FloatTensor(obs[f'server_{i}']).unsqueeze(0) 
                     for i in range(self.num_agents)}
            
            while step_count < self.env.max_steps:
                # 所有智能体选择动作
                actions = {}
                for i, agent in enumerate(self.agents):
                    state_tensor = states[f'server_{i}']
                    action = agent.select_action(state_tensor, add_noise=True)
                    actions[f'server_{i}'] = action
                
                # 执行动作
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # 转换为tensor
                next_states = {f'server_{i}': torch.FloatTensor(next_obs[f'server_{i}']).unsqueeze(0) 
                              for i in range(self.num_agents)}
                
                # 存储经验
                for i, agent in enumerate(self.agents):
                    state = states[f'server_{i}'].squeeze(0).numpy()
                    action = actions[f'server_{i}']
                    reward = rewards[f'server_{i}']
                    next_state = next_states[f'server_{i}'].squeeze(0).numpy()
                    done = terminated[f'server_{i}'] or truncated[f'server_{i}']
                    
                    agent.add_experience(state, action, reward, next_state, done)
                
                # 检查是否结束
                if all(terminated.values()) or all(truncated.values()):
                    break
                
                # 更新状态
                states = next_states
                total_reward += sum(rewards.values())
                step_count += 1
                
                # 检查是否结束
                if all(terminated.values()) or all(truncated.values()):
                    break
            
            # 在每个时间步都尝试训练（如果缓冲区有足够的数据）
            if len(self.agents[0].replay_buffer) >= self.agents[0].batch_size:
                # 从经验回放缓冲区采样
                indices = random.sample(range(len(self.agents[0].replay_buffer)), self.agents[0].batch_size)
                batch = [self.agents[0].replay_buffer[i] for i in indices]
                
                # 解包批次数据
                batch_states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
                batch_actions = torch.FloatTensor(np.array([exp[1] for exp in batch]))
                batch_rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).unsqueeze(1)
                batch_next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
                batch_dones = torch.BoolTensor(np.array([exp[4] for exp in batch])).unsqueeze(1)
                
                # 对每个智能体分别训练
                for i, agent in enumerate(self.agents):
                    agent.train_on_batch(
                        batch_states, batch_actions, batch_rewards, 
                        batch_next_states, batch_dones, self.agents
                    )
            
            episode_rewards.append(total_reward)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                training_logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
        
        training_logger.info("训练完成!")
        training_logger.info(f"平均奖励: {np.mean(all_rewards):.2f}")
        training_logger.info(f"最终奖励: {all_rewards[-1]:.2f}")
        
        return all_rewards
    
    def save_models(self, model_dir: str):
        """保存所有模型"""
        os.makedirs(model_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            filepath = os.path.join(model_dir, f"server_{i}_agent.pth")
            agent.save_model(filepath)
            training_logger.info(f"保存模型: {filepath}")
    
    def load_models(self, model_dir: str):
        """加载所有模型"""
        for i, agent in enumerate(self.agents):
            filepath = os.path.join(model_dir, f"server_{i}_agent.pth")
            if os.path.exists(filepath):
                agent.load_model(filepath)
                training_logger.info(f"加载模型: {filepath}")


def main():
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
    
    # 算法配置（优化版）
    algo_config = {
        'actor_lr': 3e-4,  # 提高学习率
        'critic_lr': 1e-3,
        'gamma': 0.99,  # 提高折扣因子
        'tau': 0.005,  # 降低软更新参数
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'entropy_coef': 0.01,  # 添加熵系数
        'grad_clip': 0.5,  # 添加梯度裁剪
        'buffer_size': 100000,
        'batch_size': 128
    }
    
    # 创建环境
    env = MultiAgentServerEnv(env_config)
    
    # 创建训练器
    trainer = MADDPGTrainer(env, algo_config)
    
    # 训练
    training_episodes = 600
    rewards = trainer.train(episodes=training_episodes)
    
    # 保存模型到maddpg文件夹
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "multi_agent_server", "maddpg")
    trainer.save_models(model_dir)
    
    training_logger.info(f"平均奖励: {np.mean(rewards):.2f}")
    training_logger.info(f"最终奖励: {rewards[-1]:.2f}")


if __name__ == "__main__":
    main()