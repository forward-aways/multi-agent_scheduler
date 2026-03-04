"""
测试物流调度模型 - 随机订单模式
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer

def main():
    print("="*80)
    print("测试物流调度模型 - 随机订单模式")
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
    
    # 重置环境
    obs, _ = env.reset()
    
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
    
    loaded_count = 0
    for agent_id, agent in trainer.agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                loaded_count += 1
            except Exception as e:
                print(f"加载模型失败 {model_path}: {str(e)}")
    
    if loaded_count == 0:
        print("警告: 没有加载到任何模型")
        return
    
    print(f"\n开始测试... ({env.max_steps} 步)")
    print("="*80)
    
    # 记录每20步的状态
    for step in range(env.max_steps):
        # 获取动作
        actions = {}
        for agent_id, agent_obs in obs.items():
            state_tensor = torch.FloatTensor(agent_obs).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = trainer.agents[agent_id].actor(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
                actions[agent_id] = action
        
        # 执行动作
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 每20步输出一次详细信息
        if (step + 1) % 20 == 0:
            print(f"步骤 {step + 1:3d}: 待处理={len(env.pending_orders):2d}, "
                  f"配送中={len(env.delivering_orders):2d}, "
                  f"已完成={env.completed_orders:2d}, "
                  f"失败={env.failed_orders:2d}")
    
    print(f"\n最终结果:")
    print(f"  待处理订单: {len(env.pending_orders)}")
    print(f"  配送中订单: {len(env.delivering_orders)}")
    print(f"  已完成订单: {env.completed_orders}")
    print(f"  失败订单: {env.failed_orders}")
    
    total_processed = env.completed_orders + env.failed_orders
    success_rate = env.completed_orders / max(total_processed, 1) * 100
    print(f"  成功率: {success_rate:.2f}%")
    
    if env.completed_orders > 0:
        print(f"\n✓ 模型表现良好！")
    else:
        print(f"\n⚠ 模型需要进一步训练或优化。")
    
    return env.completed_orders, env.failed_orders

if __name__ == "__main__":
    main()
