"""
测试物流环境观测空间与训练代码的一致性
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.logistics.train_mappo_logistics import MAPPOTrainer

def main():
    print("="*80)
    print("物流环境观测空间测试")
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
        'max_steps': 200
    }
    
    # 初始化环境
    print("\n1. 初始化环境...")
    env = MultiAgentLogisticsEnv(env_config)
    print(f"   仓库数量: {env.num_warehouses}")
    print(f"   车辆数量: {env.num_vehicles}")
    print(f"   最大待处理订单: {env.max_pending_orders}")
    
    # 打印环境定义的观测空间
    print("\n2. 环境定义的观测空间:")
    for agent_id, obs_space in env.observation_spaces.items():
        print(f"   {agent_id}: shape={obs_space.shape}, dtype={obs_space.dtype}")
    
    # 获取实际观测
    print("\n3. 获取实际观测...")
    obs, _ = env.reset()
    for agent_id, observation in obs.items():
        print(f"   {agent_id}: shape={observation.shape}, dtype={observation.dtype}")
    
    # 初始化训练器
    print("\n4. 初始化训练器...")
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
    
    # 打印训练器中智能体的状态维度
    print("\n5. 训练器中智能体的状态维度:")
    for agent_id, agent in trainer.agents.items():
        print(f"   {agent_id}: state_dim={agent.state_dim}, action_dim={agent.action_dim}")
    
    # 对比维度
    print("\n6. 维度一致性检查:")
    all_match = True
    for agent_id in env.observation_spaces.keys():
        env_shape = env.observation_spaces[agent_id].shape[0]
        agent_dim = trainer.agents[agent_id].state_dim
        
        match = env_shape == agent_dim
        status = "✓ 匹配" if match else "✗ 不匹配"
        print(f"   {agent_id}: 环境{env_shape} vs 训练{agent_dim} - {status}")
        
        if not match:
            all_match = False
    
    print("\n" + "="*80)
    if all_match:
        print("结果: ✓ 所有观测空间维度一致！")
    else:
        print("结果: ✗ 存在维度不匹配，需要修复！")
    print("="*80)

if __name__ == "__main__":
    main()
