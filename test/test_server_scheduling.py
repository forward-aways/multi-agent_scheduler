"""
测试服务器调度任务
加载训练好的模型，分析服务器调度行为
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_server_env import MultiAgentServerEnv


class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


def load_server_models(num_servers, state_dim, action_dim, model_dir):
    """加载服务器模型"""
    models = {}
    for i in range(num_servers):
        model = Actor(state_dim, action_dim, hidden_dim=256)
        model_path = Path(model_dir) / f'server_{i}_agent.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        # 加载actor部分
        model.load_state_dict(checkpoint['actor_state_dict'])
        model.eval()
        models[f'server_{i}'] = model
    return models


def print_server_status(env, step):
    """打印服务器状态信息"""
    print(f"\n{'='*70}")
    print(f"步骤 {step}: 服务器状态分析")
    print(f"{'='*70}")
    
    # 待处理任务
    print(f"\n待处理任务数: {len(env.pending_tasks)}")
    if env.pending_tasks:
        print(f"待处理任务列表:")
        for i, task in enumerate(env.pending_tasks[:5]):  # 只显示前5个
            print(f"  任务{i+1}: CPU={task['cpu_req']:.1f}, 内存={task['memory_req']:.1f}, "
                  f"优先级={task['priority']}, 持续时间={task['duration']}")
        if len(env.pending_tasks) > 5:
            print(f"  ... 还有 {len(env.pending_tasks)-5} 个任务")
    
    # 服务器状态
    print(f"\n服务器状态:")
    for i in range(env.num_servers):
        cpu_usage = env.server_cpu_usage[i]
        memory_usage = env.server_memory_usage[i]
        task_count = len(env.server_current_tasks[i])
        cpu_util = cpu_usage / env.server_cpu_capacity
        mem_util = memory_usage / env.server_memory_capacity
        
        print(f"  服务器{i}: CPU={cpu_usage:.1f}/{env.server_cpu_capacity:.1f} ({cpu_util*100:.1f}%), "
              f"内存={memory_usage:.1f}/{env.server_memory_capacity:.1f} ({mem_util*100:.1f}%), "
              f"任务数={task_count}")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  完成任务数: {env.completed_tasks}")
    print(f"  丢弃任务数: {env.dropped_tasks}")
    print(f"  当前步骤: {env.current_step}/{env.max_steps}")
    
    # 计算负载均衡
    cpu_utils = [env.server_cpu_usage[i] / env.server_cpu_capacity for i in range(env.num_servers)]
    mem_utils = [env.server_memory_usage[i] / env.server_memory_capacity for i in range(env.num_servers)]
    
    print(f"\n负载均衡:")
    print(f"  CPU利用率: {[f'{u*100:.1f}%' for u in cpu_utils]}")
    print(f"  内存利用率: {[f'{u*100:.1f}%' for u in mem_utils]}")
    print(f"  CPU标准差: {np.std(cpu_utils)*100:.2f}%")
    print(f"  内存标准差: {np.std(mem_utils)*100:.2f}%")


def analyze_server_scheduling(env, models, num_steps=100):
    """分析服务器调度行为"""
    print(f"\n{'='*70}")
    print(f"开始分析服务器调度行为")
    print(f"{'='*70}")
    
    obs, _ = env.reset()
    
    total_reward = 0
    step_rewards = []
    
    # 手动添加一些任务
    print(f"\n手动添加任务...")
    tasks_to_add = [
        {'cpu_req': 20.0, 'memory_req': 30.0, 'priority': 5, 'duration': 15, 'type': 'compute'},
        {'cpu_req': 15.0, 'memory_req': 25.0, 'priority': 3, 'duration': 10, 'type': 'compute'},
        {'cpu_req': 25.0, 'memory_req': 35.0, 'priority': 4, 'duration': 12, 'type': 'io'},
        {'cpu_req': 10.0, 'memory_req': 15.0, 'priority': 2, 'duration': 8, 'type': 'compute'},
        {'cpu_req': 30.0, 'memory_req': 40.0, 'priority': 5, 'duration': 20, 'type': 'io'},
        {'cpu_req': 18.0, 'memory_req': 28.0, 'priority': 3, 'duration': 11, 'type': 'compute'},
        {'cpu_req': 22.0, 'memory_req': 32.0, 'priority': 4, 'duration': 14, 'type': 'io'},
        {'cpu_req': 12.0, 'memory_req': 18.0, 'priority': 2, 'duration': 9, 'type': 'compute'},
        {'cpu_req': 28.0, 'memory_req': 38.0, 'priority': 5, 'duration': 18, 'type': 'io'},
        {'cpu_req': 16.0, 'memory_req': 24.0, 'priority': 3, 'duration': 10, 'type': 'compute'},
    ]
    
    for i, task_params in enumerate(tasks_to_add):
        env.add_task(task_params)
        print(f"  添加任务{i+1}: CPU={task_params['cpu_req']}, 内存={task_params['memory_req']}, "
              f"优先级={task_params['priority']}, 持续时间={task_params['duration']}")
    
    for step in range(num_steps):
        # 获取模型动作
        actions = {}
        for agent_id, model in models.items():
            state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            with torch.no_grad():
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=-1)
                actions[agent_id] = action.item()
        
        # 执行动作
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_reward += step_reward
        step_rewards.append(step_reward)
        
        # 每10步打印一次状态信息
        if step % 10 == 0:
            print_server_status(env, step)
            print(f"\n动作选择:")
            for agent_id, action in actions.items():
                server_idx = int(agent_id.split('_')[1])
                action_names = ['接受任务', '拒绝任务', '优先处理']
                print(f"  {agent_id}: {action_names[action]}")
            print(f"步骤奖励: {step_reward:.2f}, 总奖励: {total_reward:.2f}")
        
        obs = next_obs
        
        if all(terminated.values()) or all(truncated.values()):
            break
    
    # 最终状态分析
    print_server_status(env, num_steps)
    
    # 统计信息
    print(f"\n{'='*70}")
    print(f"统计信息")
    print(f"{'='*70}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {np.mean(step_rewards):.2f}")
    print(f"最大奖励: {np.max(step_rewards):.2f}")
    print(f"最小奖励: {np.min(step_rewards):.2f}")
    print(f"完成任务数: {env.completed_tasks}")
    print(f"丢弃任务数: {env.dropped_tasks}")
    print(f"任务完成率: {env.completed_tasks/(env.completed_tasks+env.dropped_tasks)*100 if (env.completed_tasks+env.dropped_tasks)>0 else 0:.2f}%")
    
    # 计算最终负载均衡
    cpu_utils = [env.server_cpu_usage[i] / env.server_cpu_capacity for i in range(env.num_servers)]
    mem_utils = [env.server_memory_usage[i] / env.server_memory_capacity for i in range(env.num_servers)]
    
    print(f"\n最终负载均衡:")
    print(f"  平均CPU利用率: {np.mean(cpu_utils)*100:.2f}%")
    print(f"  平均内存利用率: {np.mean(mem_utils)*100:.2f}%")
    print(f"  CPU利用率标准差: {np.std(cpu_utils)*100:.2f}%")
    print(f"  内存利用率标准差: {np.std(mem_utils)*100:.2f}%")
    
    # 分析每个服务器的任务处理情况
    print(f"\n各服务器任务处理情况:")
    for i in range(env.num_servers):
        print(f"  服务器{i}: 任务数={len(env.server_current_tasks[i])}, "
              f"CPU={env.server_cpu_usage[i]:.1f}, 内存={env.server_memory_usage[i]:.1f}")


def main():
    """主函数"""
    print("="*70)
    print("测试服务器调度任务")
    print("="*70)
    
    # 环境配置
    config = {
        'num_servers': 5,
        'server_cpu_capacity': 100.0,
        'server_memory_capacity': 100.0,
        'server_max_tasks': 10,
        'task_generation_rate': 3,
        'max_pending_tasks': 20,
        'max_steps': 200,
        'manual_task_mode': True  # 手动任务模式
    }
    
    # 创建环境
    env = MultiAgentServerEnv(config)
    
    # 获取观测空间维度
    obs, _ = env.reset()
    state_dim = obs['server_0'].shape[0]
    action_dim = 3  # 3个离散动作：接受任务、拒绝任务、优先处理
    
    print(f"\n环境信息:")
    print(f"  服务器数量: {env.num_servers}")
    print(f"  服务器CPU容量: {env.server_cpu_capacity}")
    print(f"  服务器内存容量: {env.server_memory_capacity}")
    print(f"  服务器最大任务数: {env.server_max_tasks}")
    print(f"  观测空间维度: {state_dim}")
    print(f"  动作空间维度: {action_dim}")
    
    # 模型路径
    model_dir = project_root / 'models' / 'multi_agent_server' / 'mappo'
    
    # 检查模型文件是否存在
    if not model_dir.exists():
        print(f"\n错误: 模型目录不存在: {model_dir}")
        print("请先训练服务器调度模型")
        return
    
    # 加载模型
    print(f"\n加载模型...")
    models = load_server_models(env.num_servers, state_dim, action_dim, model_dir)
    print(f"已加载 {len(models)} 个服务器模型")
    
    # 分析服务器调度行为
    analyze_server_scheduling(env, models, num_steps=100)
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
