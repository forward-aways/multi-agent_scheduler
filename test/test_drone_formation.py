"""
测试无人机队形保持任务
加载训练好的模型，分析队形保持行为
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
    """Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
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
        model.load_state_dict(checkpoint['actor_state_dict'])
        model.eval()
        models[f'drone_{i}'] = model
    return models


def print_drone_positions(env, step):
    """打印无人机位置信息"""
    print(f"\n{'='*70}")
    print(f"步骤 {step}: 无人机位置分析")
    print(f"{'='*70}")
    
    print(f"\n领航机: 无人机{env.leader_drone_idx}")
    print(f"起点: {env.formation_start}")
    print(f"终点: {env.formation_end}")
    print(f"队形类型: {env.formation_type}")
    
    print(f"\n无人机位置:")
    for i in range(env.num_drones):
        pos = env.drone_positions[i]
        vel = env.drone_velocities[i]
        battery = env.drone_batteries[i]
        
        # 计算到终点的距离
        distance_to_end = np.linalg.norm(pos - env.formation_end)
        
        print(f"  无人机{i}: 位置=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"速度=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}], "
              f"电量={battery:.1f}%, 距离终点={distance_to_end:.2f}")
    
    # 计算队形误差
    leader_pos = env.drone_positions[env.leader_drone_idx]
    total_formation_error = 0.0
    for i in range(env.num_drones):
        if i != env.leader_drone_idx:
            expected_pos = leader_pos + env.formation_offsets[i]
            actual_pos = env.drone_positions[i]
            distance = np.linalg.norm(actual_pos - expected_pos)
            total_formation_error += distance
    
    avg_formation_error = total_formation_error / (env.num_drones - 1) if env.num_drones > 1 else 0.0
    
    print(f"\n队形误差:")
    print(f"  平均队形误差: {avg_formation_error:.2f}")
    print(f"  队形保持: {'✅ 良好' if avg_formation_error < 5.0 else '⚠️ 一般' if avg_formation_error < 10.0 else '❌ 差'}")
    
    # 任务完成状态
    print(f"\n任务状态:")
    print(f"  任务完成: {'✅ 是' if env.task_completed else '❌ 否'}")
    print(f"  当前步骤: {env.current_step}/{env.max_steps}")


def analyze_formation_behavior(env, models, num_steps=100):
    """分析队形保持行为"""
    print(f"\n{'='*70}")
    print(f"开始分析无人机队形保持行为")
    print(f"{'='*70}")
    
    obs, _ = env.reset()
    
    total_reward = 0
    step_rewards = []
    formation_errors = []
    
    for step in range(num_steps):
        # 获取模型动作
        actions = {}
        for agent_id, model in models.items():
            state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                # 模型输出是连续的3D速度
                action_3d = model(state_tensor).squeeze(0).numpy()
                # 限制速度范围
                action_3d = np.clip(action_3d, -env.max_speed, env.max_speed)
                actions[agent_id] = action_3d
        
        # 执行动作
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        step_rewards.append(step_reward)
        
        # 计算队形误差
        leader_pos = env.drone_positions[env.leader_drone_idx]
        total_formation_error = 0.0
        for i in range(env.num_drones):
            if i != env.leader_drone_idx:
                expected_pos = leader_pos + env.formation_offsets[i]
                actual_pos = env.drone_positions[i]
                distance = np.linalg.norm(actual_pos - expected_pos)
                total_formation_error += distance
        
        avg_formation_error = total_formation_error / (env.num_drones - 1) if env.num_drones > 1 else 0.0
        formation_errors.append(avg_formation_error)
        
        # 每10步打印一次位置信息
        if step % 10 == 0:
            print_drone_positions(env, step)
            print(f"\n动作选择:")
            for agent_id, action in actions.items():
                drone_idx = int(agent_id.split('_')[1])
                print(f"  {agent_id}: 速度=[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]")
            print(f"步骤奖励: {step_reward:.2f}, 总奖励: {total_reward:.2f}")
        
        obs = next_obs
        
        if env.task_completed:
            print(f"\n{'='*70}")
            print(f"任务完成！在步骤 {step} 完成队形保持任务")
            print(f"{'='*70}")
            break
        
        if all(terminated.values()) or all(truncated.values()):
            break
    
    # 最终状态分析
    print_drone_positions(env, num_steps)
    
    # 统计信息
    print(f"\n{'='*70}")
    print(f"统计信息")
    print(f"{'='*70}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {np.mean(step_rewards):.2f}")
    print(f"最大奖励: {np.max(step_rewards):.2f}")
    print(f"最小奖励: {np.min(step_rewards):.2f}")
    print(f"任务完成: {'✅ 是' if env.task_completed else '❌ 否'}")
    
    # 队形误差统计
    print(f"\n队形误差统计:")
    print(f"  平均队形误差: {np.mean(formation_errors):.2f}")
    print(f"  最小队形误差: {np.min(formation_errors):.2f}")
    print(f"  最大队形误差: {np.max(formation_errors):.2f}")
    print(f"  队形误差标准差: {np.std(formation_errors):.2f}")
    
    # 最终距离终点的距离
    print(f"\n最终距离终点:")
    for i in range(env.num_drones):
        distance_to_end = np.linalg.norm(env.drone_positions[i] - env.formation_end)
        print(f"  无人机{i}: {distance_to_end:.2f}")
    
    avg_distance = np.mean([np.linalg.norm(env.drone_positions[i] - env.formation_end) 
                           for i in range(env.num_drones)])
    print(f"  平均距离: {avg_distance:.2f}")


def main():
    """主函数"""
    print("="*70)
    print("测试无人机队形保持任务")
    print("="*70)
    
    # 环境配置
    config = {
        'num_drones': 3,
        'space_size': [100.0, 100.0, 50.0],
        'max_speed': 2.0,
        'battery_capacity': 100.0,
        'max_steps': 200,
        'task_type': 'formation',
        'formation_type': 'triangle',
        'observation_mode': 'full'
    }
    
    # 创建环境
    env = MultiAgentDroneEnv(config)
    
    # 获取观测空间维度
    obs, _ = env.reset()
    state_dim = obs['drone_0'].shape[0]
    action_dim = 3  # 3D连续动作
    
    print(f"\n环境信息:")
    print(f"  无人机数量: {env.num_drones}")
    print(f"  空间大小: {env.space_size}")
    print(f"  最大速度: {env.max_speed}")
    print(f"  电池容量: {env.battery_capacity}")
    print(f"  观测空间维度: {state_dim}")
    print(f"  动作空间维度: {action_dim}")
    
    # 模型路径
    model_dir = project_root / 'models' / 'multi_agent_drone' / 'mappo' / 'formation' / 'triangle'
    
    # 检查模型文件是否存在
    if not model_dir.exists():
        print(f"\n错误: 模型目录不存在: {model_dir}")
        print("请先训练队形保持模型")
        return
    
    # 加载模型
    print(f"\n加载模型...")
    models = load_drone_models(env.num_drones, state_dim, action_dim, model_dir)
    print(f"已加载 {len(models)} 个无人机模型")
    
    # 分析队形保持行为
    analyze_formation_behavior(env, models, num_steps=100)
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
