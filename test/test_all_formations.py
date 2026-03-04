"""
测试所有队形模型
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path

from environments.multi_agent_drone_env import MultiAgentDroneEnv

# 导入DroneMAPPOAgent
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'drone'))
from train_mappo_formation import DroneMAPPOAgent


def load_drone_models(num_drones, state_dim, action_dim, model_dir):
    """加载无人机模型"""
    models = {}
    for i in range(num_drones):
        agent = DroneMAPPOAgent(i, state_dim, action_dim, {
            'actor_lr': 1e-4,
            'critic_lr': 5e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.15,
            'entropy_coef': 0.08,
            'value_coef': 0.5,
            'ppo_epochs': 15,
            'mini_batch_size': 32,
        })
        
        model_path = model_dir / f"drone_{i}_agent.pth"
        if model_path.exists():
            agent.load(model_path)
            models[f'drone_{i}'] = agent
    
    return models


def test_formation(formation_type):
    """测试指定队形"""
    print(f"\n{'='*80}")
    print(f"测试队形: {formation_type}")
    print(f"{'='*80}")
    
    project_root = Path(__file__).parent.parent
    
    env_config = {
        'num_drones': 3,
        'max_speed': 2.0,
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'formation',
        'formation_type': formation_type,
        'max_steps': 200
    }
    
    env = MultiAgentDroneEnv(env_config)
    
    # 加载模型
    model_dir = Path(project_root) / "models" / "multi_agent_drone" / "mappo" / "formation" / formation_type
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return None
    
    print(f"从 {model_dir} 加载模型...")
    
    state_dim = env.observation_spaces['drone_0'].shape[0]
    action_dim = env.num_actions
    
    models = load_drone_models(3, state_dim, action_dim, model_dir)
    if len(models) < 3:
        print(f"❌ 模型加载失败，只加载了 {len(models)} 个模型")
        return None
    
    print(f"✓ 已加载 {len(models)} 个无人机模型")
    
    # 开始测试
    obs, _ = env.reset()
    total_reward = 0
    
    print(f"\n开始测试... ({env.max_steps} 步)")
    print("="*80)
    
    for step in range(env.max_steps):
        actions = {}
        for agent_id, model in models.items():
            state = obs[agent_id]
            with torch.no_grad():
                action_idx = model.select_action(state, training=False)
                actions[agent_id] = action_idx
        
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        if (step + 1) % 20 == 0:
            leader_pos = env.drone_positions[env.leader_drone_idx]
            total_formation_error = 0.0
            for i in range(env.num_drones):
                if i != env.leader_drone_idx:
                    expected_pos = leader_pos + env.formation_offsets[i]
                    actual_pos = env.drone_positions[i]
                    distance = np.linalg.norm(actual_pos - expected_pos)
                    total_formation_error += distance
            
            avg_formation_error = total_formation_error / (env.num_drones - 1)
            
            print(f"步骤 {step + 1:3d}: 奖励={step_reward:7.2f}, 总奖励={total_reward:8.2f}, "
                  f"队形误差={avg_formation_error:6.2f}")
        
        obs = next_obs
        
        if env.task_completed:
            print(f"\n✅ 任务完成！在步骤 {step} 完成队形保持任务")
            break
    
    # 最终统计
    print(f"\n{'='*80}")
    print(f"统计信息 - {formation_type}")
    print(f"{'='*80}")
    print(f"总奖励：{total_reward:.2f}")
    print(f"平均奖励：{total_reward / env.max_steps:.2f}")
    
    leader_pos = env.drone_positions[env.leader_drone_idx]
    total_formation_error = 0.0
    for i in range(env.num_drones):
        if i != env.leader_drone_idx:
            expected_pos = leader_pos + env.formation_offsets[i]
            actual_pos = env.drone_positions[i]
            distance = np.linalg.norm(actual_pos - expected_pos)
            total_formation_error += distance
    
    avg_formation_error = total_formation_error / (env.num_drones - 1)
    print(f"最终队形误差：{avg_formation_error:.2f}")
    print(f"队形保持：{'✅ 良好' if avg_formation_error < 5.0 else '⚠️ 一般' if avg_formation_error < 10.0 else '❌ 差'}")
    
    return {
        'formation': formation_type,
        'total_reward': total_reward,
        'avg_reward': total_reward / env.max_steps,
        'formation_error': avg_formation_error
    }


def main():
    """主函数"""
    print("\n" + "="*80)
    print("测试所有无人机队形模型")
    print("="*80)
    
    formations = ['triangle', 'v_shape', 'line']
    results = []
    
    for formation in formations:
        result = test_formation(formation)
        if result:
            results.append(result)
    
    # 汇总
    print(f"\n{'='*80}")
    print("汇总对比")
    print(f"{'='*80}")
    print(f"{'队形':<15} {'总奖励':<12} {'平均奖励':<12} {'队形误差':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['formation']:<15} {r['total_reward']:<12.2f} {r['avg_reward']:<12.2f} {r['formation_error']:<12.2f}")
    
    print(f"\n{'='*80}")
    print("所有测试完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
