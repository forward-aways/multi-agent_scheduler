"""
测试物流调度模型 - 对比手动控制
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer

def test_with_model(env, trainer):
    """使用模型测试"""
    print("\n" + "="*60)
    print("使用训练好的模型测试")
    print("="*60)
    
    obs, _ = env.reset()
    
    for step in range(50):
        actions = {}
        for agent_id, agent_obs in obs.items():
            state_tensor = torch.FloatTensor(agent_obs).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = trainer.agents[agent_id].actor(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
                actions[agent_id] = action
        
        obs, rewards, done, truncated, info = env.step(actions)
        
        if (step + 1) % 10 == 0:
            print(f"步骤 {step + 1}: 待处理={len(env.pending_orders)}, "
                  f"配送中={len(env.delivering_orders)}, "
                  f"已完成={env.completed_orders}, 失败={env.failed_orders}")

def test_with_manual_control(env):
    """使用手动控制测试"""
    print("\n" + "="*60)
    print("使用手动控制测试")
    print("="*60)
    
    obs, _ = env.reset()
    
    for step in range(50):
        actions = {}
        
        # 仓库动作：分配订单
        for i in range(env.num_warehouses):
            actions[f'warehouse_{i}'] = 0
        
        # 车辆动作：手动控制
        if step < 2:
            for i in range(env.num_vehicles):
                actions[f'vehicle_{i}'] = 0  # 去仓库
        elif step < 4:
            for i in range(env.num_vehicles):
                actions[f'vehicle_{i}'] = 1  # 去配送
        else:
            for i in range(env.num_vehicles):
                actions[f'vehicle_{i}'] = 3  # 等待
        
        obs, rewards, done, truncated, info = env.step(actions)
        
        if (step + 1) % 10 == 0:
            print(f"步骤 {step + 1}: 待处理={len(env.pending_orders)}, "
                  f"配送中={len(env.delivering_orders)}, "
                  f"已完成={env.completed_orders}, 失败={env.failed_orders}")

def main():
    print("="*80)
    print("物流调度模型测试 - 对比手动控制")
    print("="*80)
    
    # 环境配置（与训练配置一致）
    env_config = {
        'num_warehouses': 3,
        'num_vehicles': 5,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,
        'order_generation_rate': 2,
        'max_pending_orders': 15,
        'map_size': [100.0, 100.0],
        'max_steps': 200,
        'manual_mode': False
    }
    
    # 初始化环境
    env = MultiAgentLogisticsEnv(env_config)
    
    # 初始化训练器（加载模型）
    trainer_config = {
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
    
    trainer = MAPPOTrainer(env, trainer_config)
    
    # 加载模型
    import os
    model_dir = os.path.join(project_root, "models", "multi_agent_logistics", "mappo")
    
    for agent_id, agent in trainer.agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
            except Exception as e:
                print(f"加载模型失败 {model_path}: {str(e)}")
    
    print("\n测试1: 使用训练好的模型")
    test_with_model(env, trainer)
    
    print("\n测试2: 使用手动控制")
    test_with_manual_control(env)

if __name__ == "__main__":
    main()
