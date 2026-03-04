"""
测试无人机协同包围任务
加载训练好的模型，分析无人机位置和包围行为
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_drone_env import MultiAgentDroneEnv


class Actor(nn.Module):
    """Actor网络（离散动作版）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)  # 输出logits，不使用Tanh
        )
        
    def forward(self, x):
        return self.network(x)


def load_drone_models(num_drones, state_dim, action_dim, model_dir):
    """加载无人机模型"""
    models = {}
    for i in range(num_drones):
        model = Actor(state_dim, action_dim, hidden_dim=256)
        model_path = Path(model_dir) / f'drone_{i}_agent.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        # 加载actor部分
        model.load_state_dict(checkpoint['actor_state_dict'])
        model.eval()
        models[f'drone_{i}'] = model
    return models


def print_drone_positions(env, step):
    """打印无人机位置信息"""
    print(f"\n{'='*60}")
    print(f"步骤 {step}: 无人机位置分析")
    print(f"{'='*60}")
    
    # 目标位置
    print(f"目标位置: [{env.target_position[0]:.2f}, {env.target_position[1]:.2f}, {env.target_position[2]:.2f}]")
    print(f"目标速度: [{env.target_velocity[0]:.2f}, {env.target_velocity[1]:.2f}, {env.target_velocity[2]:.2f}]")
    print(f"包围半径: {env.encirclement_radius}")
    
    # 无人机位置
    print(f"\n无人机位置:")
    for i in range(env.num_drones):
        pos = env.drone_positions[i]
        vel = env.drone_velocities[i]
        battery = env.drone_batteries[i]
        
        # 计算到目标的距离
        distance = np.linalg.norm(pos - env.target_position)
        in_radius = "✓ 在包围半径内" if distance < env.encirclement_radius else "✗ 在包围半径外"
        
        print(f"  无人机{i}: 位置=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"速度=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}], "
              f"电量={battery:.1f}, 距离目标={distance:.2f}, {in_radius}")
    
    # 包围状态
    print(f"\n包围状态:")
    print(f"  包围时间: {env.encirclement_time}/50")
    print(f"  包围步数: {env.encirclement_steps}")
    print(f"  包围成功: {'是' if env.encirclement_success else '否'}")
    
    # 计算包围角度分布
    print(f"\n包围角度分布:")
    for i in range(env.num_drones):
        pos = env.drone_positions[i]
        direction = pos - env.target_position
        angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        print(f"  无人机{i}: 角度={angle:.2f}°")
    
    # 计算无人机之间的角度差异
    angles = []
    for i in range(env.num_drones):
        pos = env.drone_positions[i]
        direction = pos - env.target_position
        angle = np.arctan2(direction[1], direction[0])
        angles.append(angle)
    
    angles.sort()
    angle_diffs = []
    for i in range(len(angles)):
        diff = angles[(i + 1) % len(angles)] - angles[i]
        if diff < 0:
            diff += 2 * np.pi
        angle_diffs.append(diff * 180 / np.pi)
    
    print(f"  角度差异: {[f'{d:.2f}°' for d in angle_diffs]}")
    print(f"  平均角度差异: {np.mean(angle_diffs):.2f}°")
    print(f"  角度差异标准差: {np.std(angle_diffs):.2f}°")


def analyze_encirclement_behavior(env, models, num_steps=100):
    """分析包围行为"""
    print(f"\n{'='*60}")
    print(f"开始分析无人机协同包围行为")
    print(f"{'='*60}")
    
    obs, _ = env.reset()
    
    total_reward = 0
    step_rewards = []
    
    for step in range(num_steps):
        # 获取模型动作（离散动作）
        actions = {}
        for agent_id, model in models.items():
            state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                # 模型输出是离散动作的概率分布，选择概率最大的动作
                logits = model(state_tensor)
                action_idx = torch.argmax(logits, dim=-1).item()
                actions[agent_id] = action_idx
        
        # 执行动作
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        step_rewards.append(step_reward)
        
        # 每10步打印一次位置信息
        if step % 10 == 0:
            print_drone_positions(env, step)
            print(f"  步骤奖励: {step_reward:.2f}, 总奖励: {total_reward:.2f}")
        
        obs = next_obs
        
        if env.encirclement_success:
            print(f"\n{'='*60}")
            print(f"包围成功！在步骤 {step} 完成包围")
            print(f"{'='*60}")
            break
        
        if all(terminated.values()) or all(truncated.values()):
            break
    
    # 最终位置分析
    print_drone_positions(env, num_steps)
    
    # 统计信息
    print(f"\n{'='*60}")
    print(f"统计信息")
    print(f"{'='*60}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {np.mean(step_rewards):.2f}")
    print(f"最大奖励: {np.max(step_rewards):.2f}")
    print(f"最小奖励: {np.min(step_rewards):.2f}")
    print(f"包围成功: {'是' if env.encirclement_success else '否'}")
    print(f"包围时间: {env.encirclement_time}/50")
    
    # 计算最终包围质量
    distances = []
    for i in range(env.num_drones):
        distance = np.linalg.norm(env.drone_positions[i] - env.target_position)
        distances.append(distance)
    
    print(f"\n最终距离目标:")
    for i, dist in enumerate(distances):
        print(f"  无人机{i}: {dist:.2f}")
    print(f"平均距离: {np.mean(distances):.2f}")
    print(f"最大距离: {np.max(distances):.2f}")


def main():
    """主函数"""
    print("="*60)
    print("测试无人机协同包围任务")
    print("="*60)
    
    # 环境配置（必须与训练时一致）
    config = {
        'num_drones': 3,
        'max_speed': 2.0,  # 必须与训练时一致
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'encirclement',
        'max_steps': 200
    }
    
    # 创建环境
    env = MultiAgentDroneEnv(config)
    
    # 获取观测空间维度
    obs, _ = env.reset()
    state_dim = obs['drone_0'].shape[0]
    action_dim = env.num_actions  # 27个离散动作
    
    print(f"\n环境信息:")
    print(f"  无人机数量: {env.num_drones}")
    print(f"  任务类型: {env.task_type}")
    print(f"  观测空间维度: {state_dim}")
    print(f"  动作空间维度: {action_dim} (离散动作)")
    print(f"  包围半径: {env.encirclement_radius}")
    print(f"  目标位置: {env.target_position}")
    print(f"  目标速度: {env.target_velocity}")
    
    # 模型路径
    model_dir = project_root / 'models' / 'multi_agent_drone' / 'mappo' / 'encirclement'
    
    # 检查模型文件是否存在
    if not model_dir.exists():
        print(f"\n错误: 模型目录不存在: {model_dir}")
        print("请先训练无人机协同包围任务模型")
        return
    
    # 加载模型
    print(f"\n加载模型...")
    models = load_drone_models(env.num_drones, state_dim, action_dim, model_dir)
    print(f"已加载 {len(models)} 个无人机模型")
    
    # 分析包围行为
    analyze_encirclement_behavior(env, models, num_steps=100)
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
