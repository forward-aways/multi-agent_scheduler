"""
调试车辆装货问题
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv

def debug_vehicle_loading():
    """调试车辆装货问题"""
    print("="*80)
    print("调试车辆装货问题")
    print("="*80)
    
    env_config = {
        'num_warehouses': 2,
        'num_vehicles': 2,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,
        'order_generation_rate': 0,
        'max_pending_orders': 10,
        'map_size': [100.0, 100.0],
        'max_steps': 100,
        'manual_mode': True
    }
    
    env = MultiAgentLogisticsEnv(env_config)
    obs, _ = env.reset()
    
    print(f"仓库位置: {env.warehouse_positions}")
    print(f"车辆初始位置: {env.vehicle_positions}")
    print(f"车辆速度: {env.vehicle_speed}")
    
    # 添加两个订单
    env.pending_orders.append([np.array([20.0, 30.0]), 5, 5])
    env.pending_orders.append([np.array([40.0, 50.0]), 5, 3])
    env.all_orders.extend([
        {'position': np.array([20.0, 30.0]), 'quantity': 5, 'priority': 5, 'status': 'pending'},
        {'position': np.array([40.0, 50.0]), 'quantity': 5, 'priority': 3, 'status': 'pending'}
    ])
    
    # 步骤1: 两个仓库都分配订单
    actions = {'warehouse_0': 0, 'warehouse_1': 0, 'vehicle_0': 3, 'vehicle_1': 3}
    obs, rewards, done, truncated, info = env.step(actions)
    
    print(f"\n步骤1后:")
    print(f"  warehouse_orders[0]: {env.warehouse_orders[0]}")
    print(f"  warehouse_orders[1]: {env.warehouse_orders[1]}")
    print(f"  delivering_orders: {[(o[0].tolist(), o[1], o[2]) for o in env.delivering_orders]}")
    
    # 步骤2-10: 车辆0去仓库0
    print(f"\n车辆0去仓库0:")
    for step in range(10):
        actions = {'warehouse_0': 3, 'warehouse_1': 3, 'vehicle_0': 0, 'vehicle_1': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        
        dist_to_wh0 = np.linalg.norm(env.vehicle_positions[0] - np.array(env.warehouse_positions[0]))
        print(f"  步骤{step+2}: 车辆0位置={env.vehicle_positions[0]}, 到仓库0距离={dist_to_wh0:.2f}, 状态={env.vehicle_status[0]}, 载货={env.vehicle_cargo[0]}")
        
        if env.vehicle_cargo[0] > 0:
            print(f"  ✅ 车辆0装货成功!")
            break
    
    # 步骤11-20: 车辆1去仓库1
    print(f"\n车辆1去仓库1:")
    for step in range(10):
        actions = {'warehouse_0': 3, 'warehouse_1': 3, 'vehicle_0': 3, 'vehicle_1': 0}
        obs, rewards, done, truncated, info = env.step(actions)
        
        dist_to_wh1 = np.linalg.norm(env.vehicle_positions[1] - np.array(env.warehouse_positions[1]))
        print(f"  步骤{step+11}: 车辆1位置={env.vehicle_positions[1]}, 到仓库1距离={dist_to_wh1:.2f}, 状态={env.vehicle_status[1]}, 载货={env.vehicle_cargo[1]}")
        
        if env.vehicle_cargo[1] > 0:
            print(f"  ✅ 车辆1装货成功!")
            break
    
    print(f"\n最终状态:")
    print(f"  车辆0: 载货={env.vehicle_cargo[0]}, 目标订单={env.vehicle_target_order_pos[0]}")
    print(f"  车辆1: 载货={env.vehicle_cargo[1]}, 目标订单={env.vehicle_target_order_pos[1]}")
    print(f"  warehouse_orders[0]: {env.warehouse_orders[0]}")
    print(f"  warehouse_orders[1]: {env.warehouse_orders[1]}")

if __name__ == "__main__":
    debug_vehicle_loading()
