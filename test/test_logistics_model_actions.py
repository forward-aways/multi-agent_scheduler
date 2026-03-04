"""
测试物流调度模型，观察多个订单的处理情况
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
    print("物流调度模型测试")
    print("="*80)
    
    # 环境配置
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
        'manual_mode': True
    }
    
    # 初始化环境
    print("\n1. 初始化环境...")
    env = MultiAgentLogisticsEnv(env_config)
    print(f"   仓库数量: {env.num_warehouses}")
    print(f"   车辆数量: {env.num_vehicles}")
    print(f"   最大待处理订单: {env.max_pending_orders}")
    
    # 先重置环境，初始化库存
    print("\n2. 重置环境，初始化库存...")
    obs, _ = env.reset()
    print(f"   仓库库存: {env.warehouse_inventory}")
    
    # 添加多个订单
    print("\n3. 添加多个订单...")
    orders_to_add = [
        {'position': np.array([20.0, 30.0]), 'quantity': 10, 'priority': 5},
        {'position': np.array([40.0, 50.0]), 'quantity': 15, 'priority': 3},
        {'position': np.array([60.0, 70.0]), 'quantity': 8, 'priority': 4},
        {'position': np.array([80.0, 20.0]), 'quantity': 12, 'priority': 2},
        {'position': np.array([30.0, 80.0]), 'quantity': 6, 'priority': 5},
    ]
    
    for i, order in enumerate(orders_to_add):
        order_list = [order['position'], order['quantity'], order['priority']]
        env.pending_orders.append(order_list)
        
        # 添加到all_orders
        order_info = {
            'position': order['position'],
            'quantity': order['quantity'],
            'priority': order['priority'],
            'status': 'pending'
        }
        env.all_orders.append(order_info)
        
        print(f"   订单{i+1}: 位置={order['position']}, 数量={order['quantity']}, 优先级={order['priority']}")
    
    print(f"\n   待处理订单数: {len(env.pending_orders)}")
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
    
    # 获取初始观测
    print("\n5. 获取初始观测...")
    obs = env._get_observations()
    
    # 打印仓库状态
    print("\n6. 仓库初始状态:")
    for i in range(env.num_warehouses):
        print(f"   仓库{i}: 库存={env.warehouse_inventory[i]:.1f}, 订单数={len(env.warehouse_orders[i])}")
    
    # 打印车辆状态
    print("\n7. 车辆初始状态:")
    for i in range(env.num_vehicles):
        status_names = {0: '空闲', 1: '去仓库', 2: '去配送', 3: '返回仓库'}
        print(f"   车辆{i}: 位置={env.vehicle_positions[i]}, 状态={status_names[env.vehicle_status[i]]}, 载货={env.vehicle_cargo[i]:.1f}")
    
    # 运行推理并打印动作
    print("\n8. 开始推理...")
    print("="*80)
    
    action_names = {
        'warehouse': {0: '分配订单', 1: '拒绝订单', 2: '调整库存'},
        'vehicle': {0: '去仓库', 1: '去配送', 2: '返回仓库', 3: '等待'}
    }
    
    for step in range(10):
        print(f"\n步骤 {step + 1}:")
        print("-"*80)
        
        # 获取观测
        obs = env._get_observations()
        
        # 转换为tensor
        states = {}
        for agent_id in obs.keys():
            states[agent_id] = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
        
        # 获取动作
        actions = {}
        for agent_id in obs.keys():
            state_tensor = states[agent_id]
            
            # 使用模型选择动作（不加噪声）
            with torch.no_grad():
                action_probs = trainer.agents[agent_id].actor(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
                actions[agent_id] = action
        
        # 打印仓库动作
        print("  仓库动作:")
        for i in range(env.num_warehouses):
            agent_id = f'warehouse_{i}'
            action = actions[agent_id]
            action_name = action_names['warehouse'][action]
            print(f"    {agent_id}: {action} ({action_name})")
        
        # 打印车辆动作
        print("  车辆动作:")
        for i in range(env.num_vehicles):
            agent_id = f'vehicle_{i}'
            action = actions[agent_id]
            action_name = action_names['vehicle'][action]
            print(f"    {agent_id}: {action} ({action_name})")
        
        # 执行动作
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 打印订单状态
        print("  订单状态:")
        for i, order_info in enumerate(env.all_orders):
            status_map = {'pending': '待处理', 'delivering': '配送中', 'completed': '已完成', 'failed': '失败'}
            status = status_map.get(order_info['status'], order_info['status'])
            print(f"    订单{i+1}: {status}")
        
        # 打印待处理订单
        print(f"  待处理订单数: {len(env.pending_orders)}")
        for i, order in enumerate(env.pending_orders):
            print(f"    待处理订单{i+1}: 位置={order[0]}, 数量={order[1]}, 优先级={order[2]}")
        
        # 打印配送中订单
        print(f"  配送中订单数: {len(env.delivering_orders)}")
        for i, order in enumerate(env.delivering_orders):
            print(f"    配送中订单{i+1}: 位置={order[0]}, 数量={order[1]}, 优先级={order[2]}")
        
        # 打印仓库状态
        print("  仓库状态:")
        for i in range(env.num_warehouses):
            print(f"    仓库{i}: 库存={env.warehouse_inventory[i]:.1f}, 订单数={len(env.warehouse_orders[i])}")
        
        # 打印车辆状态
        print("  车辆状态:")
        for i in range(env.num_vehicles):
            status_names = {0: '空闲', 1: '去仓库', 2: '去配送', 3: '返回仓库'}
            target_pos = env.vehicle_target_order_pos[i]
            target_str = f", 目标={target_pos}" if target_pos is not None else ""
            print(f"    车辆{i}: 位置={env.vehicle_positions[i]}, 状态={status_names[env.vehicle_status[i]]}, 载货={env.vehicle_cargo[i]:.1f}{target_str}")
        
        # 检查是否所有订单都已处理
        pending_count = sum(1 for o in env.all_orders if o['status'] == 'pending')
        delivering_count = sum(1 for o in env.all_orders if o['status'] == 'delivering')
        completed_count = sum(1 for o in env.all_orders if o['status'] == 'completed')
        failed_count = sum(1 for o in env.all_orders if o['status'] == 'failed')
        
        print(f"  统计: 待处理={pending_count}, 配送中={delivering_count}, 已完成={completed_count}, 失败={failed_count}")
        
        if pending_count == 0 and delivering_count == 0:
            print("\n  所有订单已处理完毕！")
            break
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)

if __name__ == "__main__":
    main()
