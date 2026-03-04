"""
测试无人机调度环境
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_drone_env import MultiAgentDroneEnv


def test_drone_inspection_mission():
    """测试无人机巡检任务"""
    print("=== 测试无人机巡检任务 ===")
    
    config = {
        'num_drones': 3,
        'max_speed': 10.0,
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'inspection',
        'num_waypoints': 4,
        'max_steps': 200
    }
    
    env = MultiAgentDroneEnv(config)
    
    obs, infos = env.reset()
    print(f"环境重置成功，无人机数量: {len(obs)}")
    print(f"观测空间: {[obs[key].shape for key in obs.keys()]}")
    print(f"检查点数量: {len(env.waypoints)}")
    print(f"检查点位置: {env.waypoints}")
    
    total_reward = 0
    for step in range(50):
        actions = {}
        for agent_id in obs.keys():
            # 随机选择动作（3D速度）
            action = np.random.uniform(-1, 1, 3) * env.max_speed
            actions[agent_id] = action
        
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        print(f"步骤 {step + 1}: 奖励={rewards}, 总奖励={total_reward:.2f}, "
              f"巡检进度={env.waypoints_visited}/{len(env.waypoints)}")
        
        obs = next_obs
        
        if all(terminated.values()) or all(truncated.values()):
            print("环境提前终止")
            break
    
    print(f"巡检任务测试完成，总奖励: {total_reward:.2f}")
    print(f"完成的检查点: {env.waypoints_visited}/{len(env.waypoints)}")


def test_drone_formation_mission():
    """测试无人机队形任务"""
    print("\n=== 测试无人机队形任务 ===")
    
    config = {
        'num_drones': 3,
        'max_speed': 10.0,
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'formation',
        'formation_type': 'triangle',
        'max_steps': 200
    }
    
    env = MultiAgentDroneEnv(config)
    
    obs, infos = env.reset()
    print(f"环境重置成功，无人机数量: {len(obs)}")
    print(f"队形类型: {env.formation_type}")
    print(f"队形目标位置: {env.formation_target}")
    
    total_reward = 0
    for step in range(50):
        actions = {}
        for agent_id in obs.keys():
            # 随机选择动作（3D速度）
            action = np.random.uniform(-1, 1, 3) * env.max_speed
            actions[agent_id] = action
        
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        print(f"步骤 {step + 1}: 奖励={rewards}, 总奖励={total_reward:.2f}, "
              f"队形误差={env.formation_error:.2f}")
        
        obs = next_obs
        
        if all(terminated.values()) or all(truncated.values()):
            print("环境提前终止")
            break
    
    print(f"队形任务测试完成，总奖励: {total_reward:.2f}")
    print(f"最终队形误差: {env.formation_error:.2f}")


def test_formation_switching():
    """测试队形切换"""
    print("\n=== 测试队形切换 ===")
    
    config = {
        'num_drones': 3,
        'max_speed': 10.0,
        'battery_capacity': 100.0,
        'payload_capacity': 5.0,
        'space_size': [100, 100, 50],
        'task_type': 'formation',
        'formation_type': 'triangle',
        'max_steps': 200
    }
    
    env = MultiAgentDroneEnv(config)
    
    formations = ['triangle', 'v_shape', 'line']
    
    for formation_type in formations:
        print(f"\n切换到队形: {formation_type}")
        env.set_formation_type(formation_type)
        obs, _ = env.reset()
        
        print(f"队形目标位置: {env.formation_target}")
        
        # 执行几步
        for step in range(5):
            actions = {}
            for agent_id in obs.keys():
                action = np.random.uniform(-1, 1, 3) * env.max_speed
                actions[agent_id] = action
            
            obs, rewards, terminated, truncated, infos = env.step(actions)
            print(f"  步骤 {step + 1}: 队形误差={env.formation_error:.2f}")


def main():
    """主函数"""
    print("测试无人机调度环境")
    
    try:
        test_drone_inspection_mission()
        test_drone_formation_mission()
        test_formation_switching()
        
        print("\n✓ 无人机调度环境测试通过!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
