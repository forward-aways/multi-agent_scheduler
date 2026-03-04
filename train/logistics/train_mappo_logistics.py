"""
多智能体物流调度训练脚本 - MAPPO算法
使用MAPPO算法训练多智能体协作的物流调度系统
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

from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
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


class LogisticsMAPPOAgent:
    """物流MAPPO智能体"""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: dict):
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
        """选择动作"""
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
        """更新网络（MAPPO算法）"""
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
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                new_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(new_values, batch_returns)
                
                # 熵正则化
                entropy = dist.entropy().mean()
                
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
    
    def __init__(self, env: MultiAgentLogisticsEnv, config: dict):
        self.env = env
        self.config = config
        self.num_warehouses = env.num_warehouses
        self.num_vehicles = env.num_vehicles
        
        # 创建智能体
        self.agents = {}
        
        # 仓库智能体
        for i in range(self.num_warehouses):
            # 计算状态维度
            local_dim = 1 + 1  # 库存+订单
            neighbor_dim = (self.num_warehouses - 1) * 2  # 邻居仓库状态
            order_dim = min(5, env.max_pending_orders) * 3  # 待处理订单信息
            state_dim = local_dim + neighbor_dim + order_dim
            action_dim = 3  # 分配订单, 拒绝订单, 调整库存
            
            training_logger.info(f"训练-仓库{i}: 本地{local_dim} + 邻居{neighbor_dim} + 订单{order_dim} = {state_dim}")
            
            agent = LogisticsMAPPOAgent(f'warehouse_{i}', state_dim, action_dim, config)
            self.agents[f'warehouse_{i}'] = agent
        
        # 车辆智能体
        for i in range(self.num_vehicles):
            # 计算状态维度
            local_dim = 2 + 1 + 1 + 1 + 1  # 位置+载货+状态+目标仓库+目标订单
            neighbor_dim = (self.num_vehicles - 1) * 3  # 邻居车辆状态
            order_dim = min(5, env.max_pending_orders) * 3  # 待处理订单信息
            state_dim = local_dim + neighbor_dim + order_dim
            action_dim = 4  # 去仓库, 去配送, 返回仓库, 等待
            
            training_logger.info(f"训练-车辆{i}: 本地{local_dim} + 邻居{neighbor_dim} + 订单{order_dim} = {state_dim}")
            
            agent = LogisticsMAPPOAgent(f'vehicle_{i}', state_dim, action_dim, config)
            self.agents[f'vehicle_{i}'] = agent
        
        # 训练统计
        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.best_model_saved = False
    
    def train(self, episodes: int = 500):
        """训练模型"""
        training_logger.info(f"开始训练物流调度任务（MAPPO），共 {episodes} 回合")
        
        for episode in range(episodes):
            # 重置环境
            observations, _ = self.env.reset()
            
            # 初始化每个智能体的经验回放缓冲区
            for agent in self.agents.values():
                agent.states = []
                agent.actions = []
                agent.log_probs = []
                agent.rewards = []
                agent.values = []
                agent.dones = []
            
            episode_reward = 0.0
            
            # 执行一个回合
            for step in range(self.env.max_steps):
                # 每个智能体选择动作
                actions = {}
                for agent_id, agent in self.agents.items():
                    state = observations[agent_id]
                    action, log_prob, value = agent.select_action(state, training=True)
                    agent.store_transition(state, action, log_prob, 0.0, value, False)
                    actions[agent_id] = action
                
                # 执行动作
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # 更新奖励
                for agent_id, reward in rewards.items():
                    episode_reward += reward
                    self.agents[agent_id].rewards[-1] = reward
                    self.agents[agent_id].dones[-1] = terminated.get(agent_id, False) or truncated.get(agent_id, False)
                
                # 检查是否结束
                if all(terminated.values()) or all(truncated.values()):
                    break
                
                observations = next_states
            
            # 更新网络
            for agent in self.agents.values():
                agent.update()
            
            # 记录奖励
            self.episode_rewards.append(episode_reward / (self.num_warehouses + self.num_vehicles))
            
            # 检查是否是最佳模型（使用滑动平均）
            if len(self.episode_rewards) >= 50:
                current_avg = np.mean(self.episode_rewards[-50:])
                if current_avg > self.best_reward:
                    self.best_reward = current_avg
                    self.save_best_model()
                    self.best_model_saved = True
                    training_logger.info(f"更新最佳模型（50回合滑动平均）: {current_avg:.2f}")
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                max_reward = np.max(self.episode_rewards)
                print(f"进度: [{episode+1}/{episodes}] ({(episode+1)/episodes*100:.1f}%) | 本回合奖励: {episode_reward/(self.num_warehouses+self.num_vehicles):.2f} | 最近10回合平均: {avg_reward:.2f} | 历史最高: {max_reward:.2f}")
                training_logger.info(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}")
        
        # 保存模型
        model_dir = self.config.get('model_dir', 'models/logistics')
        os.makedirs(model_dir, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            model_path = os.path.join(model_dir, f'{agent_id}_agent.pth')
            agent.save(model_path)
            training_logger.info(f"保存物流模型: {model_path}")
        
        # 打印训练结果
        avg_reward = np.mean(self.episode_rewards)
        final_reward = self.episode_rewards[-1]
        max_reward = np.max(self.episode_rewards)
        final_avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else final_reward
        
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"最终奖励: {final_reward:.2f}")
        print(f"最高奖励: {max_reward:.2f}")
        print(f"最近50回合平均: {final_avg_reward:.2f}")
        print(f"最佳滑动平均: {self.best_reward:.2f}")
        print(f"最佳模型已保存: {self.best_model_saved}")
        print(f"{'='*60}\n")
        
        training_logger.info(f"物流调度任务训练完成!")
        training_logger.info(f"平均奖励: {avg_reward:.2f}")
        training_logger.info(f"最终奖励: {final_reward:.2f}")
        training_logger.info(f"最高奖励: {max_reward:.2f}")
        training_logger.info(f"最近50回合平均: {final_avg_reward:.2f}")
        training_logger.info(f"最佳滑动平均: {self.best_reward:.2f}")
        training_logger.info(f"最佳模型已保存: {self.best_model_saved}")
        
        return self.episode_rewards
    
    def save_best_model(self):
        """保存最佳模型"""
        model_dir = self.config.get('model_dir', 'models/logistics')
        best_model_dir = os.path.join(model_dir, 'best')
        os.makedirs(best_model_dir, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            model_path = os.path.join(best_model_dir, f'{agent_id}_agent.pth')
            agent.save(model_path)
            training_logger.info(f"保存最佳模型: {model_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练物流调度任务')
    parser.add_argument('--warehouses', type=int, default=3,
                       help='仓库数量')
    parser.add_argument('--vehicles', type=int, default=5,
                       help='车辆数量')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='训练回合数')
    args = parser.parse_args()
    
    env_config = {
        'num_warehouses': args.warehouses,
        'num_vehicles': args.vehicles,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,
        'order_generation_rate': 2,
        'max_pending_orders': 15,
        'map_size': [100.0, 100.0],
        'max_steps': 200
    }
    
    env = MultiAgentLogisticsEnv(env_config)
    
    config = {
        'model_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "models", "multi_agent_logistics", "mappo"),
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
    
    trainer = MAPPOTrainer(env, config)
    
    training_episodes = args.episodes
    rewards = trainer.train(episodes=training_episodes)
    
    print(f"物流调度任务训练完成，模型已保存到 {config['model_dir']}")


if __name__ == "__main__":
    main()