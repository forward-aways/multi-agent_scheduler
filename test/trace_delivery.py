"""
追踪单个订单的完整配送流程
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer

class TraceEnv(MultiAgentLogisticsEnv):
    """带追踪功能的环境"""
    
    def _update_vehicle_positions(self):
        """重写方法，添加详细追踪"""
        for i in range(self.num_vehicles):
            if self.vehicle_status[i] == 2:  # 去配送
                if self.vehicle_target_order_pos[i] is not None:
                    target_pos = np.array(self.vehicle_target_order_pos[i])
                    current_pos = self.vehicle_positions[i]
                    direction = target_pos - current_pos
                    distance = np.linalg.norm(direction)
                    
                    print(f"  [车辆{i}] 位置={current_pos.round(2)}, 目标={target_pos.round(2)}, 距离={distance:.2f}, 速度={self.vehicle_speed}")
                    
                    if distance > 0:
                        self.vehicle_positions[i] += (direction / distance) * self.vehicle_speed
                    
                    # 检查是否到达配送点
                    if distance <= self.vehicle_speed:
                        print(f"  [车辆{i}] 🎯 到达配送点！")
                        self.vehicle_positions[i] = target_pos
                        self.vehicle_cargo[i] = 0
                        
                        # 尝试从 delivering_orders 中移除
                        removed = False
                        for idx, order in enumerate(self.delivering_orders):
                            if np.array_equal(order[0], target_pos):
                                self.delivering_orders.pop(idx)
                                print(f"  [车辆{i}] ✅ 从 delivering_orders 移除订单")
                                removed = True
                                break
                        if not removed:
                            print(f"  [车辆{i}] ❌ 未在 delivering_orders 中找到订单！")
                            print(f"    目标位置: {target_pos}")
                            print(f"    delivering_orders: {[(o[0].round(2), o[1]) for o in self.delivering_orders[:5]]}")
                        
                        # 更新 all_orders
                        updated = False
                        for order_info in self.all_orders:
                            if np.array_equal(order_info['position'], target_pos):
                                print(f"  [车辆{i}] 找到 all_orders 中的订单，当前状态: {order_info['status']}")
                                if order_info['status'] == 'delivering':
                                    order_info['status'] = 'completed'
                                    self.completed_orders += 1
                                    print(f"  [车辆{i}] ✅ 订单标记为已完成！")
                                    updated = True
                                else:
                                    print(f"  [车辆{i}] ❌ 订单状态不是'delivering'，无法完成！")
                                break
                        if not updated:
                            print(f"  [车辆{i}] ❌ 未在 all_orders 中找到匹配的订单！")
                        
                        self.vehicle_target_order_pos[i] = None
                        self.vehicle_status[i] = 0
                        print(f"  [车辆{i}] 状态变为空闲")
            else:
                # 其他状态使用原方法
                super()._update_vehicle_positions()

def trace():
    print("="*80)
    print("订单配送追踪")
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
    env = TraceEnv(env_config)
    
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
    
    print(f"\n初始订单: {[(o[0].round(2).tolist(), o[1], o[2]) for o in env.pending_orders]}")
    
    for step in range(50):
        actions = {}
        for agent_id, agent in trainer.agents.items():
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            actions[agent_id] = action
        
        # 找到正在配送的车辆
        delivering_vehicles = [i for i in range(env.num_vehicles) if env.vehicle_status[i] == 2]
        
        if delivering_vehicles:
            print(f"\n步骤 {step}:")
            print(f"  配送中车辆: {delivering_vehicles}")
            print(f"  delivering_orders 数量: {len(env.delivering_orders)}")
        
        obs, _, _, _, _ = env.step(actions)
        
        if env.completed_orders > 0:
            print(f"\n🎉 第 {step} 步完成第一个订单！")
            break
    
    print(f"\n最终: 完成={env.completed_orders}, 失败={env.failed_orders}")

if __name__ == "__main__":
    trace()
