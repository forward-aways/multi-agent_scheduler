"""
检查订单状态流转
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer

def check():
    print("="*80)
    print("检查订单状态")
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
        'actor_lr': 3e-4, 'critic_lr': 1e-3, 'gamma': 0.99,
        'gae_lambda': 0.95, 'clip_epsilon': 0.2,
        'entropy_coef': 0.01, 'value_coef': 0.5,
        'ppo_epochs': 10, 'mini_batch_size': 64
    }
    
    trainer = MAPPOTrainer(env, trainer_config)
    
    # 加载模型
    model_dir = "models/multi_agent_logistics/mappo/best"
    for agent_id, agent in trainer.agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
        if os.path.exists(model_path):
            agent.load(model_path)
    
    obs, _ = env.reset()
    
    print(f"\n初始 all_orders 状态:")
    for i, order in enumerate(env.all_orders[:3]):
        print(f"  订单{i}: 位置={order['position'].round(2)}, 数量={order['quantity']}, 状态={order['status']}")
    
    # 运行几步
    for step in range(20):
        actions = {}
        for agent_id, agent in trainer.agents.items():
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            actions[agent_id] = action
        
        obs, _, _, _, _ = env.step(actions)
        
        # 检查是否有订单被接受
        if step == 5:
            print(f"\n步骤 {step} 后的 all_orders 状态:")
            delivering = [o for o in env.all_orders if o['status'] == 'delivering']
            pending = [o for o in env.all_orders if o['status'] == 'pending']
            print(f"  配送中: {len(delivering)}, 待处理: {len(pending)}")
            for i, order in enumerate(delivering[:3]):
                print(f"    配送订单{i}: 位置={order['position'].round(2)}, 状态={order['status']}")
    
    print(f"\n最终 all_orders 状态:")
    status_count = {}
    for order in env.all_orders:
        status = order['status']
        status_count[status] = status_count.get(status, 0) + 1
    for status, count in status_count.items():
        print(f"  {status}: {count}")
    
    print(f"\n统计:")
    print(f"  all_orders 总数: {len(env.all_orders)}")
    print(f"  pending_orders: {len(env.pending_orders)}")
    print(f"  delivering_orders: {len(env.delivering_orders)}")
    print(f"  completed_orders: {env.completed_orders}")
    print(f"  failed_orders: {env.failed_orders}")

if __name__ == "__main__":
    check()
