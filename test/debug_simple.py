"""
简化版物流调度调试
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer

def debug():
    print("="*80)
    print("简化版物流调度调试")
    print("="*80)
    
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
    env = MultiAgentLogisticsEnv(env_config)
    
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
    model_dir = "models/multi_agent_logistics/mappo/best"
    for agent_id, agent in trainer.agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"✓ {agent_id}")
    
    obs, _ = env.reset()
    
    print(f"\n初始: 待处理={len(env.pending_orders)}, 配送中={len(env.delivering_orders)}, 完成={env.completed_orders}")
    
    for step in range(100):
        actions = {}
        for agent_id, agent in trainer.agents.items():
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            actions[agent_id] = action
        
        obs, rewards, _, _, _ = env.step(actions)
        
        if step % 10 == 0:
            print(f"步骤{step:3d}: 待处理={len(env.pending_orders):2d}, 配送中={len(env.delivering_orders):3d}, 完成={env.completed_orders:2d}, 失败={env.failed_orders:2d}")
        
        if env.completed_orders > 0:
            print(f"\n🎉 第 {step} 步完成第一个订单！")
            break
    
    print(f"\n最终: 完成={env.completed_orders}, 失败={env.failed_orders}")
    print("="*80)

if __name__ == "__main__":
    debug()
