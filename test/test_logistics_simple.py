"""
简单测试物流调度环境 - 验证订单完成
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv

def main():
    print("="*80)
    print("简单测试物流调度环境 - 验证订单完成")
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
        'max_steps': 50,
        'manual_mode': True
    }
    
    # 初始化环境
    env = MultiAgentLogisticsEnv(env_config)
    
    # 重置环境
    obs, _ = env.reset()
    
    # 添加订单
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
    
    print(f"初始状态:")
    print(f"  待处理订单数: {len(env.pending_orders)}")
    print(f"  所有订单状态: {[order['status'] for order in env.all_orders]}")
    
    # 手动控制动作
    for step in range(10):
        print(f"\n步骤 {step + 1}:")
        print("-"*60)
        
        actions = {}
        
        # 仓库动作：分配订单（动作0）
        for i in range(env.num_warehouses):
            actions[f'warehouse_{i}'] = 0
        
        # 车辆动作：
        if step == 0:
            # 车辆0去仓库，车辆1等待
            actions['vehicle_0'] = 0
            actions['vehicle_1'] = 3
            print(f"  动作: 车辆0去仓库(0), 车辆1等待(3)")
        elif step == 1:
            # 车辆0去仓库，车辆1等待
            actions['vehicle_0'] = 0
            actions['vehicle_1'] = 3
            print(f"  动作: 车辆0去仓库(0), 车辆1等待(3)")
        elif step == 2:
            # 车辆0去配送，车辆1等待
            actions['vehicle_0'] = 1
            actions['vehicle_1'] = 3
            print(f"  动作: 车辆0去配送(1), 车辆1等待(3)")
        elif step == 3:
            # 车辆0去配送，车辆1等待
            actions['vehicle_0'] = 1
            actions['vehicle_1'] = 3
            print(f"  动作: 车辆0去配送(1), 车辆1等待(3)")
        elif step == 4:
            # 车辆0返回仓库，车辆1等待
            actions['vehicle_0'] = 2
            actions['vehicle_1'] = 3
            print(f"  动作: 车辆0返回仓库(2), 车辆1等待(3)")
        else:
            actions['vehicle_0'] = 3
            actions['vehicle_1'] = 3
            print(f"  动作: 车辆0等待(3), 车辆1等待(3)")
        
        # 执行动作
        obs, rewards, done, truncated, info = env.step(actions)
        
        # 显示状态
        print(f"  状态: 待处理={len(env.pending_orders)}, 配送中={len(env.delivering_orders)}, 已完成={env.completed_orders}")
        print(f"  订单状态: {[order['status'] for order in env.all_orders]}")
        print(f"  车辆0: 位置={env.vehicle_positions[0]}, 状态={env.vehicle_status[0]}, 载货={env.vehicle_cargo[0]:.1f}")
        print(f"  车辆1: 位置={env.vehicle_positions[1]}, 状态={env.vehicle_status[1]}, 载货={env.vehicle_cargo[1]:.1f}")
        
        # 检查是否所有订单都已完成
        completed_count = sum(1 for o in env.all_orders if o['status'] == 'completed')
        if completed_count == len(env.all_orders):
            print(f"\n  ✓ 所有订单已完成！")
            break

if __name__ == "__main__":
    main()
