"""
验证无人机环境初始化是否正确
"""
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_drone_env import MultiAgentDroneEnv

def test_formation_init():
    """测试编队任务初始化"""
    print("="*80)
    print("测试编队任务环境初始化")
    print("="*80)
    
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
    obs, _ = env.reset()
    
    print(f"\n编队类型: {env.formation_type}")
    print(f"队形偏移量:")
    for i, offset in enumerate(env.formation_offsets):
        print(f"  无人机{i}: {offset}")
    
    print(f"\n起点: {env.formation_start}")
    print(f"终点: {env.formation_end}")
    
    print(f"\n实际初始化位置:")
    for i in range(env.num_drones):
        print(f"  无人机{i}: {env.drone_positions[i]}")
    
    # 验证位置是否正确
    print(f"\n验证:")
    leader_pos = env.drone_positions[env.leader_drone_idx]
    expected_leader_pos = env.formation_start
    leader_correct = np.allclose(leader_pos, expected_leader_pos)
    print(f"  领航机位置正确: {leader_correct}")
    if not leader_correct:
        print(f"    期望: {expected_leader_pos}")
        print(f"    实际: {leader_pos}")
    
    all_correct = leader_correct
    for i in range(env.num_drones):
        if i != env.leader_drone_idx:
            expected_pos = env.formation_start + env.formation_offsets[i]
            actual_pos = env.drone_positions[i]
            correct = np.allclose(actual_pos, expected_pos)
            all_correct = all_correct and correct
            print(f"  无人机{i}位置正确: {correct}")
            if not correct:
                print(f"    期望: {expected_pos}")
                print(f"    实际: {actual_pos}")
    
    # 计算初始队形误差
    print(f"\n初始队形误差: {env.formation_error:.4f}")
    
    if all_correct and env.formation_error < 0.01:
        print("\n✅ 编队初始化正确！")
    else:
        print("\n❌ 编队初始化有问题！")
    
    return all_correct

def test_encirclement_init():
    """测试围捕任务初始化"""
    print("\n" + "="*80)
    print("测试围捕任务环境初始化")
    print("="*80)
    
    env_config = {
        'num_drones': 3,
        'max_speed': 2.0,
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'encirclement',
        'max_steps': 200
    }
    
    env = MultiAgentDroneEnv(env_config)
    obs, _ = env.reset()
    
    print(f"\n目标位置: {env.target_position}")
    print(f"包围半径: {env.encirclement_radius}")
    
    print(f"\n实际初始化位置:")
    for i in range(env.num_drones):
        dist = np.linalg.norm(env.drone_positions[i] - env.target_position)
        print(f"  无人机{i}: {env.drone_positions[i]}, 距离目标: {dist:.2f}")
    
    # 验证距离
    print(f"\n验证:")
    expected_radius = env.encirclement_radius * 1.5
    all_correct = True
    for i in range(env.num_drones):
        dist = np.linalg.norm(env.drone_positions[i][:2] - env.target_position[:2])  # XY平面距离
        correct = abs(dist - expected_radius) < 1.0  # 允许1.0的误差
        all_correct = all_correct and correct
        print(f"  无人机{i}距离正确: {correct} (期望: {expected_radius:.2f}, 实际: {dist:.2f})")
    
    if all_correct:
        print("\n✅ 围捕初始化正确！")
    else:
        print("\n❌ 围捕初始化有问题！")
    
    return all_correct

def test_random_action():
    """测试随机动作，看无人机是否会正常移动"""
    print("\n" + "="*80)
    print("测试随机动作（验证环境step是否正常）")
    print("="*80)
    
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
    obs, _ = env.reset()
    
    print(f"\n初始位置:")
    initial_positions = env.drone_positions.copy()
    for i in range(env.num_drones):
        print(f"  无人机{i}: {env.drone_positions[i]}")
    
    # 执行10步随机动作
    print(f"\n执行10步随机动作...")
    for step in range(10):
        actions = {}
        for i in range(env.num_drones):
            # 随机动作，范围在 [-max_speed, max_speed]
            action = np.random.uniform(-env.max_speed, env.max_speed, 3)
            actions[f'drone_{i}'] = action
        
        obs, rewards, _, _, _ = env.step(actions)
    
    print(f"\n10步后位置:")
    for i in range(env.num_drones):
        movement = np.linalg.norm(env.drone_positions[i] - initial_positions[i])
        print(f"  无人机{i}: {env.drone_positions[i]}, 移动距离: {movement:.2f}")
    
    print(f"\n队形误差: {env.formation_error:.4f}")
    print("\n✅ 环境step正常！")

if __name__ == "__main__":
    formation_ok = test_formation_init()
    encirclement_ok = test_encirclement_init()
    test_random_action()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print(f"编队初始化: {'✅ 通过' if formation_ok else '❌ 失败'}")
    print(f"围捕初始化: {'✅ 通过' if encirclement_ok else '❌ 失败'}")
