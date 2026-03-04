"""
详细测试物流调度模型，验证订单状态更新逻辑
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer, LogisticsMAPPOAgent

def main():
    print("="*80)
    print("详细物流调度模型测试 - 订单状态验证")
    print("="*80)
    
    # 环境配置
    env_config = {
        'num_warehouses': 2,
        'num_vehicles': 2,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,  # 提高速度以加快测试
        'order_generation_rate': 1,
        'max_pending_orders': 10,
        'map_size': [100.0, 100.0],
        'max_steps': 100,
        'manual_mode': True
    }
    
    # 初始化环境
    print("\n1. 初始化环境...")
    env = MultiAgentLogisticsEnv(env_config)
    
    # 重置环境
    print("\n2. 重置环境...")
    obs, _ = env.reset()
    
    # 添加少量订单用于详细测试
    print("\n3. 添加测试订单...")
    test_orders = [
        {'position': np.array([20.0, 30.0]), 'quantity': 5, 'priority': 5},
        {'position': np.array([40.0, 50.0]), 'quantity': 5, 'priority': 3},
    ]
    
    for i, order in enumerate(test_orders):
        order_list = [order['position'], order['quantity'], order['priority']]
        env.pending_orders.append(order_list)
        
        order_info = {
            'position': order['position'],
            'quantity': order['quantity'],
            'priority': order['priority'],
            'status': 'pending'
        }
        env.all_orders.append(order_info)
        
        print(f"   订单{i+1}: 位置={order['position']}, 数量={order['quantity']}, 优先级={order['priority']}")
    
    print(f"   待处理订单数: {len(env.pending_orders)}")
    print(f"   所有订单数: {len(env.all_orders)}")
    
    # 初始化训练器（加载模型）
    print("\n4. 加载模型...")
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
    
    # 尝试加载模型
    import os
    model_dir = os.path.join(project_root, "models", "multi_agent_logistics", "mappo")
    
    model_loaded = False
    for agent_id, agent in trainer.agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"   已加载模型: {model_path}")
                model_loaded = True
            except Exception as e:
                print(f"   加载模型失败 {model_path}: {str(e)}")
    
    if not model_loaded:
        print("   警告: 未找到训练好的模型，将使用随机初始化的模型")
    
    print("\n5. 开始详细测试...")
    print("="*80)
    
    for step in range(20):  # 增加步数以便观察订单完成
        print(f"\n步骤 {step + 1}:")
        print("-"*60)
        
        # 获取观测
        obs = env._get_observations()
        
        # 获取动作
        actions = {}
        for agent_id in obs.keys():
            state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            
            # 使用模型选择动作（不加噪声）
            with torch.no_grad():
                action_probs = trainer.agents[agent_id].actor(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
                actions[agent_id] = action
        
        # 执行动作前的状态
        print(f"  执行前 - 待处理: {len(env.pending_orders)}, 配送中: {len(env.delivering_orders)}, 已完成: {env.completed_orders}")
        print(f"  所有订单状态: {[order['status'] for order in env.all_orders]}")
        
        # 执行动作
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 执行后的状态
        print(f"  执行后 - 待处理: {len(env.pending_orders)}, 配送中: {len(env.delivering_orders)}, 已完成: {env.completed_orders}")
        print(f"  所有订单状态: {[order['status'] for order in env.all_orders]}")
        
        # 显示车辆状态
        for i in range(env.num_vehicles):
            pos = env.vehicle_positions[i]
            status_names = {0: '空闲', 1: '去仓库', 2: '去配送', 3: '返回仓库'}
            status = status_names[env.vehicle_status[i]]
            target_pos = env.vehicle_target_order_pos[i]
            target_str = f", 目标={target_pos}" if target_pos is not None else ""
            print(f"    车辆{i}: 位置={pos}, 状态={status}, 载货={env.vehicle_cargo[i]:.1f}{target_str}")
        
        # 检查是否所有订单都已处理
        pending_count = sum(1 for o in env.all_orders if o['status'] == 'pending')
        delivering_count = sum(1 for o in env.all_orders if o['status'] == 'delivering')
        completed_count = sum(1 for o in env.all_orders if o['status'] == 'completed')
        failed_count = sum(1 for o in env.all_orders if o['status'] == 'failed')
        
        print(f"  统计: 待处理={pending_count}, 配送中={delivering_count}, 已完成={completed_count}, 失败={failed_count}")
        
        if completed_count == len(env.all_orders):
            print("\n  所有订单已完成！")
            break
    
    print("\n" + "="*80)
    print("详细测试完成")
    print("="*80)

if __name__ == "__main__":
    main()