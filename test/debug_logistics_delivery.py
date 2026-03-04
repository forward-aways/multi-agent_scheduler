"""
物流调度配送调试脚本 - 详细追踪车辆配送行为
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer
import logging

# 设置详细日志
logging.basicConfig(level=logging.DEBUG)
env_logger = logging.getLogger('logistics_env')
env_logger.setLevel(logging.DEBUG)

def debug_delivery():
    """调试配送过程"""
    print("="*80)
    print("物流调度配送调试")
    print("="*80)
    
    # 创建环境
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
    env = MultiAgentLogisticsEnv(env_config)
    
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
    
    # 加载最佳模型
    model_dir = "models/multi_agent_logistics/mappo/best"
    
    loaded_count = 0
    for agent_id, agent in trainer.agents.items():
        model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"✓ 加载模型: {agent_id}")
                loaded_count += 1
            except Exception as e:
                print(f"✗ 加载模型失败 {agent_id}: {str(e)}")
    
    if loaded_count == 0:
        print("✗ 没有加载任何模型")
        return
    
    # 重置环境
    observations, _ = env.reset()
    
    print("\n" + "="*80)
    print("初始状态:")
    print(f"  仓库位置: {env.warehouse_positions}")
    print(f"  车辆位置: {[p.tolist() for p in env.vehicle_positions]}")
    print(f"  待处理订单: {len(env.pending_orders)}")
    print(f"  订单详情: {[(o[0].tolist(), o[1]) for o in env.pending_orders[:3]]}...")
    print("="*80)
    
    # 运行几步，详细追踪
    for step in range(30):
        print(f"\n{'='*80}")
        print(f"步骤 {step + 1}")
        print(f"{'='*80}")
        
        # 收集动作
        actions = {}
        for agent_id, agent in trainer.agents.items():
            obs = observations[agent_id]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            actions[agent_id] = action
        
        # 执行动作前记录状态
        print(f"\n执行动作前:")
        print(f"  车辆状态: {env.vehicle_status} (0=空闲, 1=前往仓库, 2=配送中, 3=返回仓库)")
        print(f"  车辆位置: {[p.tolist() for p in env.vehicle_positions]}")
        print(f"  车辆载货: {env.vehicle_cargo}")
        target_positions = []
        for i in range(env.num_vehicles):
            if env.vehicle_target_order_pos[i] is not None:
                target_positions.append(env.vehicle_target_order_pos[i].tolist())
            elif env.vehicle_status[i] == 1 or env.vehicle_status[i] == 3:
                wh_idx = env.vehicle_target_warehouse[i]
                target_positions.append(env.warehouse_positions[wh_idx])
            else:
                target_positions.append(None)
        print(f"  车辆目标: {target_positions}")
        print(f"  配送中订单数: {len(env.delivering_orders)}")
        print(f"  已完成订单: {env.completed_orders}")
        
        # 执行动作
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 执行动作后记录状态
        print(f"\n执行动作后:")
        print(f"  车辆状态: {env.vehicle_status}")
        print(f"  车辆位置: {[p.tolist() for p in env.vehicle_positions]}")
        print(f"  车辆载货: {env.vehicle_cargo}")
        print(f"  配送中订单数: {len(env.delivering_orders)}")
        print(f"  已完成订单: {env.completed_orders}")
        
        # 显示每个智能体的动作
        print(f"\n智能体动作:")
        for agent_id, action in actions.items():
            if 'warehouse' in agent_id:
                action_names = ['拒绝', '接受']
                action_str = action_names[action] if action < len(action_names) else f"未知({action})"
                print(f"  {agent_id}: {action_str} ({action})")
            else:
                action_names = ['空闲', '取货', '返回仓库']
                action_str = action_names[action] if action < len(action_names) else f"未知({action})"
                print(f"  {agent_id}: {action_str} ({action})")
        
        # 显示奖励
        print(f"\n奖励:")
        for agent_id, reward in rewards.items():
            print(f"  {agent_id}: {reward:.3f}")
        
        if env.completed_orders > 0:
            print(f"\n🎉 订单已完成！总计完成: {env.completed_orders}")
            
        if step >= 10 and env.completed_orders == 0:
            print(f"\n⚠️ 警告: 已经运行 {step+1} 步，但没有订单完成")
            print("  可能的问题:")
            print("  - 车辆没有正确取货")
            print("  - 车辆没有移动到配送点")
            print("  - 配送完成逻辑有问题")
    
    print("\n" + "="*80)
    print("调试结束")
    print(f"  最终完成订单: {env.completed_orders}")
    print(f"  失败订单: {env.failed_orders}")
    print(f"  配送中订单: {len(env.delivering_orders)}")
    print("="*80)

if __name__ == "__main__":
    debug_delivery()
