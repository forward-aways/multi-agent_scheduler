"""
测试物流调度环境的观测维度
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv

env_config = {
    'num_warehouses': 3,
    'num_vehicles': 5,
    'warehouse_capacity': 100,
    'vehicle_capacity': 20,
    'vehicle_speed': 5.0,
    'order_generation_rate': 2,
    'max_pending_orders': 15,
    'map_size': [100.0, 100.0],
    'max_steps': 200
}

env = MultiAgentLogisticsEnv(env_config)
observations, _ = env.reset()

print("环境观测维度：")
for agent_id, obs in observations.items():
    print(f"{agent_id}: {obs.shape}")

print("\n观测空间定义：")
for agent_id, obs_space in env.observation_spaces.items():
    print(f"{agent_id}: {obs_space.shape}")

print("\n待处理订单数：", len(env.pending_orders))
print("车辆状态：", env.vehicle_status)
print("车辆载货：", env.vehicle_cargo)
print("仓库库存：", env.warehouse_inventory)
print("仓库订单数：", [len(orders) for orders in env.warehouse_orders])