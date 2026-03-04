"""
调试配送中订单积压问题
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
    print("调试配送中订单积压问题")
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
        'max_steps': 100,
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
    
    print(f"\n初始状态:")
    print(f"  车辆位置: {[p.round(2).tolist() for p in env.vehicle_positions]}")
    print(f"  仓库位置: {env.warehouse_positions}")
    
    delivery_attempts = 0
    delivery_success = 0
    
    for step in range(100):
        actions = {}
        for agent_id, agent in trainer.agents.items():
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            actions[agent_id] = action
        
        # 记录配送中车辆
        delivering_before = [i for i in range(env.num_vehicles) if env.vehicle_status[i] == 2]
        
        obs, rewards, _, _, _ = env.step(actions)
        
        # 检查是否有车辆完成配送
        for i in range(env.num_vehicles):
            if env.vehicle_status[i] == 0 and i in delivering_before:
                delivery_attempts += 1
                # 检查是否真的完成了（载货变为0）
                if env.vehicle_cargo[i] == 0:
                    delivery_success += 1
        
        if step % 20 == 0:
            print(f"\n步骤 {step}:")
            print(f"  车辆状态: {env.vehicle_status} (0=空闲, 1=去仓库, 2=配送中, 3=返回)")
            print(f"  车辆位置: {[p.round(2).tolist() for p in env.vehicle_positions]}")
            print(f"  车辆载货: {env.vehicle_cargo}")
            print(f"  配送中订单: {len(env.delivering_orders)}")
            print(f"  已完成订单: {env.completed_orders}")
            
            # 显示配送中车辆的目标
            for i in range(env.num_vehicles):
                if env.vehicle_status[i] == 2:
                    target = env.vehicle_target_order_pos[i]
                    if target is not None:
                        dist = np.linalg.norm(env.vehicle_positions[i] - target)
                        print(f"  车辆{i}: 目标={target.round(2)}, 距离={dist:.2f}")
    
    print(f"\n{'='*80}")
    print(f"配送统计:")
    print(f"  尝试配送: {delivery_attempts}")
    print(f"  成功完成: {delivery_success}")
    print(f"  最终配送中: {len(env.delivering_orders)}")
    print(f"  最终已完成: {env.completed_orders}")
    print(f"  最终失败: {env.failed_orders}")

if __name__ == "__main__":
    debug()
