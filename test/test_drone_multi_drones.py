"""
测试多无人机（超过3个）队形任务
验证队形偏移量动态扩展功能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from environments.multi_agent_drone_env import MultiAgentDroneEnv

def test_formation_with_5_drones():
    """测试5个无人机的队形任务"""
    print("="*80)
    print("测试: 5个无人机的队形任务")
    print("="*80)
    
    env_config = {
        'num_drones': 5,  # 5个无人机
        'max_speed': 2.0,
        'battery_capacity': 100.0,
        'space_size': [100, 100, 50],
        'task_type': 'formation',
        'formation_type': 'line',
        'max_steps': 50,
    }
    
    try:
        env = MultiAgentDroneEnv(env_config)
        obs, _ = env.reset()
        
        print(f"无人机数量: {env.num_drones}")
        print(f"队形偏移量数量: {len(env.formation_offsets)}")
        print(f"队形偏移量: {[offset.tolist() for offset in env.formation_offsets]}")
        print(f"无人机位置:\n{env.drone_positions}")
        
        # 运行几步
        for step in range(10):
            actions = {}
            for i in range(env.num_drones):
                # 随机动作
                actions[f'drone_{i}'] = np.random.randint(0, 27)
            
            obs, rewards, done, truncated, info = env.step(actions)
            
            if any(done.values()):
                break
        
        print(f"\n运行10步后状态正常")
        print(f"队形误差: {env.formation_error:.2f}")
        
        # 检查是否有越界错误
        success = len(env.formation_offsets) >= env.num_drones
        print(f"\n{'✅ 测试通过' if success else '❌ 测试失败'}")
        return success
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_formation_with_10_drones():
    """测试10个无人机的队形任务"""
    print("\n" + "="*80)
    print("测试: 10个无人机的队形任务")
    print("="*80)
    
    env_config = {
        'num_drones': 10,  # 10个无人机
        'max_speed': 2.0,
        'battery_capacity': 100.0,
        'space_size': [200, 200, 50],
        'task_type': 'formation',
        'formation_type': 'triangle',
        'max_steps': 50,
    }
    
    try:
        env = MultiAgentDroneEnv(env_config)
        obs, _ = env.reset()
        
        print(f"无人机数量: {env.num_drones}")
        print(f"队形偏移量数量: {len(env.formation_offsets)}")
        print(f"队形偏移量: {[offset.tolist() for offset in env.formation_offsets]}")
        
        # 运行几步
        for step in range(10):
            actions = {}
            for i in range(env.num_drones):
                actions[f'drone_{i}'] = np.random.randint(0, 27)
            
            obs, rewards, done, truncated, info = env.step(actions)
            
            if any(done.values()):
                break
        
        print(f"\n运行10步后状态正常")
        print(f"队形误差: {env.formation_error:.2f}")
        
        success = len(env.formation_offsets) >= env.num_drones
        print(f"\n{'✅ 测试通过' if success else '❌ 测试失败'}")
        return success
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inspection_with_5_drones():
    """测试5个无人机的巡检任务"""
    print("\n" + "="*80)
    print("测试: 5个无人机的巡检任务")
    print("="*80)
    
    env_config = {
        'num_drones': 5,
        'max_speed': 2.0,
        'battery_capacity': 100.0,
        'space_size': [100, 100, 50],
        'task_type': 'inspection',
        'num_waypoints': 4,
        'max_steps': 50,
    }
    
    try:
        env = MultiAgentDroneEnv(env_config)
        obs, _ = env.reset()
        
        print(f"无人机数量: {env.num_drones}")
        print(f"巡检路径点: {len(env.waypoints)}")
        print(f"无人机位置:\n{env.drone_positions}")
        
        # 运行几步
        for step in range(10):
            actions = {}
            for i in range(env.num_drones):
                actions[f'drone_{i}'] = np.random.randint(0, 27)
            
            obs, rewards, done, truncated, info = env.step(actions)
            
            if any(done.values()):
                break
        
        print(f"\n运行10步后状态正常")
        print(f"已访问路径点: {env.waypoints_visited}")
        
        print(f"\n✅ 测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    results = []
    results.append(("5无人机队形", test_formation_with_5_drones()))
    results.append(("10无人机队形", test_formation_with_10_drones()))
    results.append(("5无人机巡检", test_inspection_with_5_drones()))
    
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\n总体: {'✅ 所有测试通过' if all_passed else '❌ 有测试失败'}")
