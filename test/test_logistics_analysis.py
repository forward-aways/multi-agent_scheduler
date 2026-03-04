"""
详细分析物流调度模型行为
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
    print("详细分析物流调度模型行为")
    print("="*80)
    
    # 环境配置
    env_config = {
        'num_warehouses': 2,
        'num_vehicles': 2,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,
        'order_generation_rate': 1,
        'max_pending_orders': 10,
        'map_size': [100.0, 100.0],
        'max_steps': 100,
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
    
    for agent_id, agent in trainer.agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"已加载模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败 {model_path}: {str(e)}")
    
    print("\n开始测试...")
    print("="*80)
    
    # 测试一轮
    obs, _ = env.reset()
    
    for step in range(50):
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
        
        # 每5步输出一次详细信息
        if (step + 1) % 5 == 0:
            print(f"\n步骤 {step + 1}:")
            print(f"  动作分布:")
            for agent_id, action in actions.items():
                action_names = {0: '分配', 1: '拒绝', 2: '调整库存'}
                if 'warehouse' in agent_id:
                    print(f"    {agent_id}: {action_names.get(action, action)}")
                else:
                    action_names = {0: '去仓库', 1: '去配送', 2: '返回仓库', 3: '等待'}
                    print(f"    {agent_id}: {action_names.get(action, action)}")
            
            print(f"  状态:")
            print(f"    待处理订单: {len(env.pending_orders)}")
            print(f"    配送中订单: {len(env.delivering_orders)}")
            print(f"    已完成: {env.completed_orders}")
            print(f"    失败: {env.failed_orders}")
            
            print(f"  车辆状态:")
            for i in range(env.num_vehicles):
                status_names = {0: '空闲', 1: '去仓库', 2: '去配送', 3: '返回仓库'}
                print(f"    车辆{i}: 位置={env.vehicle_positions[i]}, "
                      f"状态={status_names.get(env.vehicle_status[i], env.vehicle_status[i])}, "
                      f"载货={env.vehicle_cargo[i]:.1f}")
            
            print(f"  仓库状态:")
            for i in range(env.num_warehouses):
                print(f"    仓库{i}: 库存={env.warehouse_inventory[i]:.1f}, "
                      f"待处理订单={len(env.warehouse_orders[i])}")

if __name__ == "__main__":
    main()
