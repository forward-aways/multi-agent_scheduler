"""
正式测试物流调度模型
加载训练好的模型并测试性能
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
    print("正式物流调度模型测试")
    print("="*80)
    
    # 环境配置（与训练配置一致）
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
    
    # 初始化环境
    print("\n1. 初始化环境...")
    env = MultiAgentLogisticsEnv(env_config)
    
    # 重置环境
    print("\n2. 重置环境...")
    obs, _ = env.reset()
    
    # 初始化训练器（加载模型）
    print("\n3. 加载模型...")
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
    
    # 加载模型
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
    
    print("\n4. 开始测试...")
    print("="*80)
    
    # 测试参数
    num_episodes = 5
    total_completed = 0
    total_failed = 0
    all_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # 重置环境
        obs, _ = env.reset()
        
        episode_rewards = {agent_id: 0.0 for agent_id in obs.keys()}
        episode_completed = 0
        episode_failed = 0
        
        for step in range(env.max_steps):
            # 获取动作
            actions = {}
            for agent_id, agent_obs in obs.items():
                state_tensor = torch.FloatTensor(agent_obs).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs = trainer.agents[agent_id].actor(state_tensor)
                    action = torch.argmax(action_probs, dim=-1).item()
                    actions[agent_id] = action
            
            # 执行动作
            obs, rewards, done, truncated, info = env.step(actions)
            
            # 累计奖励
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            # 统计订单完成情况
            current_completed = env.completed_orders
            current_failed = env.failed_orders
            
            if step > 0:
                episode_completed = current_completed
                episode_failed = current_failed
            
            # 每10步输出一次进度
            if (step + 1) % 10 == 0:
                print(f"  步骤 {step + 1:3d}: 待处理={len(env.pending_orders)}, "
                      f"配送中={len(env.delivering_orders)}, "
                      f"已完成={current_completed}, "
                      f"失败={current_failed}")
        
        # Episode结束统计
        print(f"\n  Episode {episode + 1} 统计:")
        print(f"    总完成订单: {episode_completed}")
        print(f"    总失败订单: {episode_failed}")
        print(f"    平均奖励:")
        for agent_id, reward in episode_rewards.items():
            print(f"    {agent_id}: {reward:.2f}")
        
        total_completed += episode_completed
        total_failed += episode_failed
        all_rewards.append(episode_rewards)
    
    # 总体统计
    print(f"\n{'='*80}")
    print("总体测试结果")
    print(f"{'='*80}")
    print(f"测试轮数: {num_episodes}")
    print(f"总完成订单: {total_completed}")
    print(f"总失败订单: {total_failed}")
    print(f"订单完成率: {total_completed / (total_completed + total_failed + 1e-6) * 100:.2f}%")
    
    # 平均奖励
    avg_rewards = {agent_id: 0.0 for agent_id in all_rewards[0].keys()}
    for rewards in all_rewards:
        for agent_id, reward in rewards.items():
            avg_rewards[agent_id] += reward
    for agent_id in avg_rewards:
        avg_rewards[agent_id] /= num_episodes
    
    print(f"\n平均奖励 (跨{num_episodes}轮):")
    for agent_id, reward in avg_rewards.items():
        print(f"  {agent_id}: {reward:.2f}")
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")
    
    return total_completed, total_failed

if __name__ == "__main__":
    main()
