"""
手动添加4个订单测试物流调度
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer

def test_manual_orders():
    print("="*80)
    print("手动添加4个订单测试")
    print("="*80)
    
    # 环境配置
    env_config = {
        'num_warehouses': 3,
        'num_vehicles': 5,
        'warehouse_capacity': 100,
        'vehicle_capacity': 20,
        'vehicle_speed': 5.0,
        'order_generation_rate': 0,  # 不自动生成订单
        'max_pending_orders': 15,
        'map_size': [100.0, 100.0],
        'max_steps': 200,
        'manual_mode': False
    }
    env = MultiAgentLogisticsEnv(env_config)
    
    # 初始化训练器
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
    
    # 重置环境
    obs, _ = env.reset()
    
    # 清空自动生成的订单
    env.pending_orders = []
    env.delivering_orders = []
    env.all_orders = []
    
    # 手动添加4个订单
    manual_orders = [
        [np.array([80.0, 80.0]), 3, 3],  # 订单1: 位置[80,80], 数量3, 优先级3
        [np.array([20.0, 80.0]), 2, 2],  # 订单2: 位置[20,80], 数量2, 优先级2
        [np.array([80.0, 20.0]), 4, 4],  # 订单3: 位置[80,20], 数量4, 优先级4
        [np.array([20.0, 20.0]), 1, 1],  # 订单4: 位置[20,20], 数量1, 优先级1
    ]
    
    print("\n手动添加4个订单:")
    for i, order in enumerate(manual_orders):
        env.pending_orders.append(order)
        env.all_orders.append({
            'position': order[0].copy(),
            'quantity': order[1],
            'priority': order[2],
            'status': 'pending',
            'assigned_warehouse': None
        })
        print(f"  订单{i+1}: 位置={order[0]}, 数量={order[1]}, 优先级={order[2]}")
    
    print(f"\n初始状态:")
    print(f"  仓库位置: {env.warehouse_positions}")
    print(f"  车辆位置: {[p.round(2).tolist() for p in env.vehicle_positions]}")
    print(f"  待处理订单: {len(env.pending_orders)}")
    
    # 运行模拟
    for step in range(100):
        # 收集动作
        actions = {}
        for agent_id, agent in trainer.agents.items():
            obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            actions[agent_id] = action
        
        # 执行动作
        obs, rewards, _, _, _ = env.step(actions)
        
        # 每20步输出状态
        if step % 20 == 0:
            print(f"\n步骤 {step}:")
            print(f"  车辆状态: {env.vehicle_status} (0=空闲, 1=去仓库, 2=配送中, 3=返回)")
            print(f"  车辆位置: {[p.round(2).tolist() for p in env.vehicle_positions]}")
            print(f"  车辆载货: {env.vehicle_cargo}")
            print(f"  待处理: {len(env.pending_orders)}, 配送中: {len(env.delivering_orders)}, 已完成: {env.completed_orders}")
            
            # 显示配送中车辆的目标
            for i in range(env.num_vehicles):
                if env.vehicle_status[i] == 2:
                    target = env.vehicle_target_order_pos[i]
                    if target is not None:
                        dist = np.linalg.norm(env.vehicle_positions[i] - target)
                        print(f"  车辆{i}: 配送目标={target.round(2)}, 距离={dist:.2f}")
        
        # 如果所有订单都完成，提前结束
        if env.completed_orders == 4:
            print(f"\n🎉 所有订单已完成！用时 {step} 步")
            break
    
    print("\n" + "="*80)
    print("最终结果:")
    print(f"  待处理订单: {len(env.pending_orders)}")
    print(f"  配送中订单: {len(env.delivering_orders)}")
    print(f"  已完成订单: {env.completed_orders}")
    print(f"  失败订单: {env.failed_orders}")
    
    if env.completed_orders == 4:
        print("  ✅ 全部完成！")
    elif env.completed_orders > 0:
        print(f"  ⚠️ 完成率: {env.completed_orders}/4 ({env.completed_orders*25}%)")
    else:
        print("  ❌ 没有完成任何订单")
    
    print("="*80)

if __name__ == "__main__":
    test_manual_orders()
