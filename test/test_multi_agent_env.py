"""
测试新的多智能体环境
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.multi_agent_server_env import MultiAgentServerEnv


def test_multi_agent_environment():
    """测试多智能体环境"""
    print("=== 测试多智能体服务器环境 ===")
    
    # 环境配置
    config = {
        'num_servers': 3,
        'server_cpu_capacity': 100.0,
        'server_memory_capacity': 100.0,
        'server_max_tasks': 8,
        'task_generation_rate': 2,
        'max_pending_tasks': 10,
        'max_steps': 50
    }
    
    # 创建环境
    env = MultiAgentServerEnv(config)
    
    # 重置环境
    obs, infos = env.reset()
    print(f"环境重置成功，智能体数量: {len(obs)}")
    print(f"观测空间: {[obs[key].shape for key in obs.keys()]}")
    
    # 执行几个步骤
    total_reward = 0
    for step in range(10):
        # 随机选择动作
        actions = {}
        for agent_id in obs.keys():
            actions[agent_id] = np.random.randint(0, 3)  # 3个动作: 0=接受, 1=拒绝, 2=优先处理
        
        # 执行动作
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # 累积奖励
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        print(f"步骤 {step + 1}: 动作={actions}, 奖励={rewards}, 总奖励={total_reward:.2f}")
        
        # 更新观测
        obs = next_obs
        
        # 检查是否结束
        if all(terminated.values()) or all(truncated.values()):
            print("环境提前终止")
            break
    
    print(f"环境测试完成，总奖励: {total_reward:.2f}")
    
    # 渲染环境状态
    env.render()
    

def main():
    """主函数"""
    print("测试新的多智能体环境")
    
    try:
        test_multi_agent_environment()
        print("\n✓ 多智能体环境测试通过!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()