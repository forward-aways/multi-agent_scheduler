"""
手动测试物流调度环境，验证订单状态更新逻辑
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv

def main():
    print("="*80)
    print("手动测试物流调度环境 - 订单状态验证")
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
    
    print("\n4. 开始手动测试...")
    print("="*80)
    
    for step in range(30):
        print(f"\n步骤 {step + 1}:")
        print("-"*60)
        
        # 手动设置动作
        actions = {}
        
        # 仓库动作：分配订单（动作0）
        for i in range(env.num_warehouses):
            actions[f'warehouse_{i}'] = 0  # 分配订单
        
        # 车辆动作：
        # 车辆0：先去仓库装货（步骤1-2），然后去配送（步骤3-4），然后返回仓库（步骤5）
        # 车辆1：等待
        if step < 2:
            actions['vehicle_0'] = 0  # 去仓库
            actions['vehicle_1'] = 3  # 等待
        elif step < 4:
            actions['vehicle_0'] = 1  # 去配送
            actions['vehicle_1'] = 3  # 等待
        elif step < 5:
            actions['vehicle_0'] = 2  # 返回仓库
            actions['vehicle_1'] = 3  # 等待
        else:
            actions['vehicle_0'] = 3  # 等待
            actions['vehicle_1'] = 3  # 等待
        
        # 执行动作前的状态
        print(f"  动作: {actions}")
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
    print("手动测试完成")
    print("="*80)

if __name__ == "__main__":
    main()
