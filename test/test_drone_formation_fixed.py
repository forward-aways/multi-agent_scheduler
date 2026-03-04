"""
测试无人机队形保持任务 - 离散动作版
与训练脚本保持一致
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_drone_env import MultiAgentDroneEnv


class Actor(nn.Module):
    """Actor 网络 - 离散动作版"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)  # 输出离散动作的概率分布
        )
        
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, state):
        """获取动作 - 离散动作，返回动作索引"""
        logits = self.network(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action


def load_drone_models(num_drones, state_dim, action_dim, model_dir):
    """加载无人机模型"""
    models = {}
    for i in range(num_drones):
        model = Actor(state_dim, action_dim, hidden_dim=256)
        model_path = Path(model_dir) / f'drone_{i}_agent.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['actor_state_dict'])
        model.eval()
        models[f'drone_{i}'] = model
    return models


def main():
    print("="*80)
    print("测试无人机队形保持任务（离散动作版）")
    print("="*80)
    
    # 环境配置（与训练一致）
    env_config = {
        'num_drones': 3,
        'max_speed': 2.0,
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'formation',
        'formation_type': 'triangle',
        'max_steps': 200
    }
    
    env = MultiAgentDroneEnv(env_config)
    
    # 加载模型
    model_dir = Path(project_root) / "models" / "multi_agent_drone" / "mappo" / "formation" / "triangle"
    print(f"\n从 {model_dir} 加载模型...")
    
    state_dim = env.observation_spaces['drone_0'].shape[0]
    action_dim = env.num_actions  # 使用离散动作数量
    
    models = load_drone_models(3, state_dim, action_dim, model_dir)
    print(f"✓ 已加载 {len(models)} 个无人机模型")
    
    # 开始测试
    obs, _ = env.reset()
    total_reward = 0
    
    print(f"\n开始测试... ({env.max_steps} 步)")
    print("="*80)
    
    for step in range(env.max_steps):
        # 获取动作 - 离散动作，返回动作索引
        actions = {}
        for agent_id, model in models.items():
            state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                # 获取离散动作索引
                action_idx = model.get_action(state_tensor).item()
                actions[agent_id] = action_idx
        
        # 执行动作
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        # 每 20 步打印一次
        if (step + 1) % 20 == 0:
            leader_pos = env.drone_positions[env.leader_drone_idx]
            total_formation_error = 0.0
            for i in range(env.num_drones):
                if i != env.leader_drone_idx:
                    expected_pos = leader_pos + env.formation_offsets[i]
                    actual_pos = env.drone_positions[i]
                    distance = np.linalg.norm(actual_pos - expected_pos)
                    total_formation_error += distance
            
            avg_formation_error = total_formation_error / (env.num_drones - 1) if env.num_drones > 1 else 0.0
            
            print(f"步骤 {step + 1:3d}: 奖励={step_reward:7.2f}, 总奖励={total_reward:8.2f}, "
                  f"队形误差={avg_formation_error:6.2f}, 任务完成={env.task_completed}")
        
        obs = next_obs
        
        if env.task_completed:
            print(f"\n✅ 任务完成！在步骤 {step} 完成队形保持任务")
            break
    
    # 最终统计
    print(f"\n{'='*80}")
    print(f"统计信息")
    print(f"{'='*80}")
    print(f"总奖励：{total_reward:.2f}")
    print(f"平均奖励：{total_reward / env.max_steps:.2f}")
    print(f"任务完成：{'✅ 是' if env.task_completed else '❌ 否'}")
    
    # 计算最终队形误差
    leader_pos = env.drone_positions[env.leader_drone_idx]
    total_formation_error = 0.0
    for i in range(env.num_drones):
        if i != env.leader_drone_idx:
            expected_pos = leader_pos + env.formation_offsets[i]
            actual_pos = env.drone_positions[i]
            distance = np.linalg.norm(actual_pos - expected_pos)
            total_formation_error += distance
    
    avg_formation_error = total_formation_error / (env.num_drones - 1) if env.num_drones > 1 else 0.0
    print(f"最终队形误差：{avg_formation_error:.2f}")
    print(f"队形保持：{'✅ 良好' if avg_formation_error < 5.0 else '⚠️ 一般' if avg_formation_error < 10.0 else '❌ 差'}")
    
    print(f"\n{'='*80}")
    print(f"测试完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
