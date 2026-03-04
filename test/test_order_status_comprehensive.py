"""
全面测试物流调度订单状态
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv

def test_order_lifecycle():
    """测试订单完整生命周期"""
    print("="*80)
    print("测试1: 订单完整生命周期")
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
    
    # 添加订单
    test_orders = [
        {'position': np.array([20.0, 30.0]), 'quantity': 5, 'priority': 5},
        {'position': np.array([40.0, 50.0]), 'quantity': 5, 'priority': 3},
    ]
    
    for order in test_orders:
        env.pending_orders.append([order['position'], order['quantity'], order['priority']])
        env.all_orders.append({
            'position': order['position'],
            'quantity': order['quantity'],
            'priority': order['priority'],
            'status': 'pending'
        })
    
    print(f"初始状态: 待处理={len(env.pending_orders)}, 配送中={len(env.delivering_orders)}, 已完成={env.completed_orders}")
    
    # 步骤1: 仓库0分配订单0，仓库1等待
    actions = {'warehouse_0': 0, 'warehouse_1': 3, 'vehicle_0': 3, 'vehicle_1': 3}
    obs, rewards, done, truncated, info = env.step(actions)
    print(f"步骤1 (仓库0分配): 待处理={len(env.pending_orders)}, 配送中={len(env.delivering_orders)}, 已完成={env.completed_orders}")
    
    # 步骤2: 仓库1分配订单1
    actions = {'warehouse_0': 3, 'warehouse_1': 0, 'vehicle_0': 3, 'vehicle_1': 3}
    obs, rewards, done, truncated, info = env.step(actions)
    print(f"步骤2 (仓库1分配): 待处理={len(env.pending_orders)}, 配送中={len(env.delivering_orders)}, 已完成={env.completed_orders}")
    
    # 步骤3-10: 车辆0去仓库0 (需要多步)
    for i in range(8):
        actions = {'warehouse_0': 3, 'warehouse_1': 3, 'vehicle_0': 0, 'vehicle_1': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        if env.vehicle_cargo[0] > 0:
            break
    print(f"步骤{i+3} (车辆0到仓库0装货): 车辆0载货={env.vehicle_cargo[0]}")
    
    # 步骤11-20: 车辆1去仓库1 (需要多步)
    for i in range(10):
        actions = {'warehouse_0': 3, 'warehouse_1': 3, 'vehicle_0': 3, 'vehicle_1': 0}
        obs, rewards, done, truncated, info = env.step(actions)
        if env.vehicle_cargo[1] > 0:
            break
    print(f"步骤{11+i} (车辆1到仓库1装货): 车辆1载货={env.vehicle_cargo[1]}")
    
    # 步骤21-30: 车辆0去配送订单0
    for i in range(10):
        actions = {'warehouse_0': 3, 'warehouse_1': 3, 'vehicle_0': 1, 'vehicle_1': 3}
        obs, rewards, done, truncated, info = env.step(actions)
        if env.vehicle_cargo[0] == 0:
            break
    print(f"步骤{21+i} (车辆0配送完成): 车辆0载货={env.vehicle_cargo[0]}, 已完成={env.completed_orders}")
    
    # 步骤31-45: 车辆1去配送订单1
    for i in range(15):
        actions = {'warehouse_0': 3, 'warehouse_1': 3, 'vehicle_0': 3, 'vehicle_1': 1}
        obs, rewards, done, truncated, info = env.step(actions)
        if env.vehicle_cargo[1] == 0:
            break
    print(f"步骤{31+i} (车辆1配送完成): 车辆1载货={env.vehicle_cargo[1]}, 已完成={env.completed_orders}")
    
    print(f"\n最终订单状态: {[o['status'] for o in env.all_orders]}")
    
    # 验证结果
    completed = sum(1 for o in env.all_orders if o['status'] == 'completed')
    print(f"\n✅ 测试通过!" if completed == 2 else f"\n❌ 测试失败! 只有{completed}/2个订单完成")
    return completed == 2

def test_order_assignment_bug():
    """测试订单分配bug - 两个仓库同时分配"""
    print("\n" + "="*80)
    print("测试2: 两个仓库同时分配订单")
    print("="*80)
    
    env_config = {
        'num_warehouses': 2,
        'num_vehicles': 1,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,
        'order_generation_rate': 0,
        'max_pending_orders': 10,
        'map_size': [100.0, 100.0],
        'max_steps': 50,
        'manual_mode': True
    }
    
    env = MultiAgentLogisticsEnv(env_config)
    obs, _ = env.reset()
    
    # 只添加1个订单
    env.pending_orders.append([np.array([30.0, 40.0]), 5, 5])
    env.all_orders.append({
        'position': np.array([30.0, 40.0]),
        'quantity': 5,
        'priority': 5,
        'status': 'pending'
    })
    
    print(f"初始: 待处理={len(env.pending_orders)}")
    
    # 两个仓库同时尝试分配同一个订单
    actions = {'warehouse_0': 0, 'warehouse_1': 0, 'vehicle_0': 3}
    obs, rewards, done, truncated, info = env.step(actions)
    
    print(f"两个仓库同时分配后: 待处理={len(env.pending_orders)}, 配送中={len(env.delivering_orders)}")
    print(f"仓库0订单数: {len(env.warehouse_orders[0])}")
    print(f"仓库1订单数: {len(env.warehouse_orders[1])}")
    
    # 检查是否有问题
    total_assigned = len(env.warehouse_orders[0]) + len(env.warehouse_orders[1])
    if total_assigned > 1:
        print(f"❌ BUG发现! 一个订单被分配到两个仓库!")
        return False
    else:
        print(f"✅ 正常，订单只分配到一个仓库")
        return True

def test_vehicle_cargo_tracking():
    """测试车辆载货追踪"""
    print("\n" + "="*80)
    print("测试3: 车辆载货追踪")
    print("="*80)
    
    env_config = {
        'num_warehouses': 1,
        'num_vehicles': 1,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,
        'order_generation_rate': 0,
        'max_pending_orders': 10,
        'map_size': [100.0, 100.0],
        'max_steps': 50,
        'manual_mode': True
    }
    
    env = MultiAgentLogisticsEnv(env_config)
    obs, _ = env.reset()
    
    # 添加订单
    env.pending_orders.append([np.array([50.0, 50.0]), 10, 5])
    env.all_orders.append({
        'position': np.array([50.0, 50.0]),
        'quantity': 10,
        'priority': 5,
        'status': 'pending'
    })
    
    print(f"初始库存: {env.warehouse_inventory[0]}")
    
    # 分配订单
    actions = {'warehouse_0': 0, 'vehicle_0': 3}
    obs, rewards, done, truncated, info = env.step(actions)
    print(f"分配后库存: {env.warehouse_inventory[0]}, warehouse_orders: {len(env.warehouse_orders[0])}")
    
    # 车辆去仓库
    actions = {'warehouse_0': 3, 'vehicle_0': 0}
    obs, rewards, done, truncated, info = env.step(actions)
    print(f"车辆到仓库后载货: {env.vehicle_cargo[0]}")
    
    # 车辆去配送
    actions = {'warehouse_0': 3, 'vehicle_0': 1}
    for i in range(10):
        obs, rewards, done, truncated, info = env.step(actions)
        if env.vehicle_cargo[0] == 0:
            print(f"步骤{i+1}: 配送完成，载货={env.vehicle_cargo[0]}")
            break
    
    print(f"最终: 已完成订单={env.completed_orders}, 订单状态={[o['status'] for o in env.all_orders]}")
    
    if env.completed_orders == 1:
        print("✅ 载货追踪正常")
        return True
    else:
        print("❌ 载货追踪有问题")
        return False

if __name__ == "__main__":
    results = []
    results.append(("订单生命周期", test_order_lifecycle()))
    results.append(("订单分配", test_order_assignment_bug()))
    results.append(("载货追踪", test_vehicle_cargo_tracking()))
    
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\n总体: {'✅ 所有测试通过' if all_passed else '❌ 有测试失败'}")
