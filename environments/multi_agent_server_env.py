"""
多智能体服务器调度环境
用于训练多智能体协作的服务器调度系统
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, List
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import ProjectLogger

env_logger = ProjectLogger('environment', log_dir='logs')


class MultiAgentServerEnv(gym.Env):
    """
    多智能体服务器调度环境
    每个服务器作为一个智能体
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境
        
        参数:
            config: 环境配置参数
        """
        super().__init__()
        
        # 服务器配置
        self.num_servers = config.get('num_servers', 5)
        self.server_cpu_capacity = config.get('server_cpu_capacity', 100.0)
        self.server_memory_capacity = config.get('server_memory_capacity', 100.0)
        self.server_max_tasks = config.get('server_max_tasks', 10)
        
        # 任务配置
        self.task_generation_rate = config.get('task_generation_rate', 3)
        self.max_pending_tasks = config.get('max_pending_tasks', 20)
        
        # 时间配置
        self.max_steps = config.get('max_steps', 200)
        
        # 服务器状态
        self.server_cpu_usage = np.zeros(self.num_servers)
        self.server_memory_usage = np.zeros(self.num_servers)
        self.server_current_tasks = [[] for _ in range(self.num_servers)]  # 每台服务器当前任务
        
        # 任务队列
        self.pending_tasks = []
        self.completed_tasks = 0
        self.dropped_tasks = 0
        
        # 时间跟踪
        self.current_step = 0
        
        # 任务模式标志
        self.manual_task_mode = config.get('manual_task_mode', False)  # 是否为手动任务模式
        
        # 为每个服务器定义动作空间和观测空间
        self.action_spaces = {}
        self.observation_spaces = {}
        
        for i in range(self.num_servers):
            # 每个服务器的动作空间：接受任务(0), 拒绝任务(1), 优先处理(2)
            self.action_spaces[f'server_{i}'] = gym.spaces.Discrete(3)
            
            # 每个服务器的观测空间
            # [CPU使用率, 内存使用率, 当前任务数, 邻居服务器状态, 待处理任务信息]
            obs_dim = 2 + 1 + (self.num_servers - 1) * 2 + min(3, self.max_pending_tasks) * 5 + 5  # 每个任务有5个特征 + 5个负载均衡特征
            self.observation_spaces[f'server_{i}'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        返回:
            初始观测和信息
        """
        super().reset(seed=seed)
        
        # 重置服务器状态
        self.server_cpu_usage = np.zeros(self.num_servers)
        self.server_memory_usage = np.zeros(self.num_servers)
        self.server_current_tasks = [[] for _ in range(self.num_servers)]
        
        # 重置任务队列
        self.pending_tasks = []
        self.completed_tasks = 0
        self.dropped_tasks = 0
        
        # 重置时间
        self.current_step = 0
        
        # 根据任务模式决定是否生成初始任务
        if not self.manual_task_mode:
            # 非手动模式下，生成初始任务
            self._generate_tasks()
        # 手动模式下，不生成初始任务，等待用户添加
        
        # 获取初始观测
        observations = self._get_observations()
        infos = {f'server_{i}': {} for i in range(self.num_servers)}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        执行一步
        
        参数:
            actions: 每个智能体的动作
        
        返回:
            (观测, 奖励, terminated, truncated, 信息)
        """
        # 处理任务分配
        rewards = {}
        for server_id, action in actions.items():
            server_idx = int(server_id.split('_')[1])
            rewards[server_id] = self._process_server_action(server_idx, action)
        
        # 处理任务执行（完成任务）
        self._process_task_completion()
        
        # 根据任务模式决定是否生成新任务
        if not self.manual_task_mode:
            # 非手动模式下，定期生成新任务
            if self.current_step % 5 == 0:  # 每5步生成一批任务
                self._generate_tasks()
        # 手动模式下，只处理已添加的任务，不自动生成新任务
        
        # 更新时间步
        self.current_step += 1
        
        # 检查终止条件
        terminated = {f'server_{i}': False for i in range(self.num_servers)}
        truncated = {f'server_{i}': self.current_step >= self.max_steps for i in range(self.num_servers)}
        
        # 获取新观测
        observations = self._get_observations()
        
        # 计算服务器利用率
        cpu_utils = [self.server_cpu_usage[i] / self.server_cpu_capacity for i in range(self.num_servers)]
        mem_utils = [self.server_memory_usage[i] / self.server_memory_capacity for i in range(self.num_servers)]
        
        # 构建info信息（用于评估）
        infos = {f'server_{i}': {} for i in range(self.num_servers)}
        
        # 在第一个智能体的info中添加全局统计信息
        # 任务总数 = 已完成 + 已丢弃（待处理的不算入总数，因为回合结束）
        total_tasks = self.completed_tasks + self.dropped_tasks
        infos['server_0'] = {
            'tasks': {
                'total': max(1, total_tasks),  # 避免除零
                'completed': self.completed_tasks,
                'failed': self.dropped_tasks,
                'pending': len(self.pending_tasks)  # 待处理任务单独记录
            },
            'agent_utilization': {
                f'server_{i}': (cpu_utils[i] + mem_utils[i]) / 2 
                for i in range(self.num_servers)
            }
        }
        
        return observations, rewards, terminated, truncated, infos
    
    def _process_server_action(self, server_idx: int, action: int) -> float:
        """
        处理服务器智能体的动作
        
        参数:
            server_idx: 服务器索引
            action: 动作 (0=接受任务, 1=拒绝任务, 2=优先处理)
            
        返回:
            奖励值
        """
        reward = 0.0
        
        # 如果有待处理任务且服务器选择接受
        if self.pending_tasks and action == 0:
            task = self.pending_tasks.pop(0)  # 取第一个任务
            
            # 检查资源是否足够
            cpu_req = task.get('cpu_req', 10)
            memory_req = task.get('memory_req', 10)
            
            if (self.server_cpu_usage[server_idx] + cpu_req <= self.server_cpu_capacity and
                self.server_memory_usage[server_idx] + memory_req <= self.server_memory_capacity and
                len(self.server_current_tasks[server_idx]) < self.server_max_tasks):
                
                # 分配任务
                self.server_cpu_usage[server_idx] += cpu_req
                self.server_memory_usage[server_idx] += memory_req
                self.server_current_tasks[server_idx].append(task)
                
                # 计算奖励：基于任务优先级
                priority = task.get('priority', 1)
                reward += priority * 2.0  # 增加优先级奖励权重，鼓励接受任务
                
                # 资源利用率奖励
                cpu_util = self.server_cpu_usage[server_idx] / self.server_cpu_capacity
                memory_util = self.server_memory_usage[server_idx] / self.server_memory_capacity
                reward += (cpu_util + memory_util) * 0.5  # 增加资源利用率奖励权重，鼓励资源使用
                
            else:
                # 资源不足，任务回到队列顶部或丢弃
                if len(self.pending_tasks) < self.max_pending_tasks:
                    self.pending_tasks.insert(0, task)  # 放回队列顶部
                else:
                    self.dropped_tasks += 1  # 丢弃任务
                    reward -= 5.0  # 增加丢弃任务惩罚，更强烈地避免任务丢失
        
        elif action == 1:  # 拒绝任务
            # 如果有任务但拒绝，可能是因为负载太高
            cpu_util = self.server_cpu_usage[server_idx] / self.server_cpu_capacity
            memory_util = self.server_memory_usage[server_idx] / self.server_memory_capacity
            
            if cpu_util > 0.8 or memory_util > 0.8:
                # 高负载下拒绝任务是合理的
                reward += 1.0  # 提高奖励以鼓励智能负载均衡
            else:
                # 低负载下拒绝任务是不好的
                reward -= 0.5  # 减少惩罚，避免过度惩罚
        
        elif action == 2:  # 优先处理
            # 如果服务器有任务，优先处理
            if self.server_current_tasks[server_idx]:
                reward += 0.3  # 优先处理奖励
        
        # 添加任务完成奖励
        # 鼓励服务器及时完成任务，提高系统吞吐量
        current_task_count = len(self.server_current_tasks[server_idx])
        prev_task_count = getattr(self, f'_prev_task_count_server_{server_idx}', current_task_count)
        task_completed_count = max(0, prev_task_count - current_task_count)  # 计算完成的任务数
        task_completion_reward = task_completed_count * 2.0  # 每完成一个任务给予奖励
        reward += task_completion_reward
        
        # 更新服务器任务计数记录
        setattr(self, f'_prev_task_count_server_{server_idx}', current_task_count)
        
        # 添加负载均衡奖励
        # 计算所有服务器的资源利用率方差，方差越小说明负载越均衡
        cpu_utils = []
        mem_utils = []
        for i in range(self.num_servers):
            cpu_utils.append(self.server_cpu_usage[i] / self.server_cpu_capacity)
            mem_utils.append(self.server_memory_usage[i] / self.server_memory_capacity)
        
        # 计算CPU和内存利用率的标准差
        cpu_std = np.std(cpu_utils)
        mem_std = np.std(mem_utils)
        
        # 标准差越小越好，所以给予负的标准差作为奖励（乘以负系数）
        # 这样可以鼓励智能体实现更均衡的负载分布
        # 使用较小的权重，以免过度影响总体奖励
        load_balance_reward = -(cpu_std + mem_std) * 0.1  # 减小权重
        reward += load_balance_reward
        
        return reward
    
    def _process_task_completion(self):
        """处理任务完成"""
        for server_idx in range(self.num_servers):
            completed_tasks = []
            remaining_tasks = []
            
            for task in self.server_current_tasks[server_idx]:
                # 模拟任务完成（基于任务持续时间）
                # 任务有一个持续时间，每一步减少计数
                task_duration = task.get('duration', 10)
                # 每一步都减少任务持续时间
                if 'remaining_duration' not in task:
                    task['remaining_duration'] = task_duration
                
                task['remaining_duration'] -= 1
                
                if task['remaining_duration'] <= 0:  # 任务完成
                    # 释放资源
                    self.server_cpu_usage[server_idx] -= task.get('cpu_req', 0)
                    self.server_memory_usage[server_idx] -= task.get('memory_req', 0)
                    self.completed_tasks += 1
                    # 注意：任务完成奖励将在服务器的动作奖励函数中体现
                    completed_tasks.append(task)
                else:
                    remaining_tasks.append(task)
            
            # 更新服务器任务列表
            self.server_current_tasks[server_idx] = remaining_tasks
    
    def add_task(self, task_params):
        """
        手动添加任务到环境
        
        参数:
            task_params: 任务参数字典
        """
        if len(self.pending_tasks) < self.max_pending_tasks:
            # 创建任务ID
            task_id = f'user_task_{len(self.pending_tasks)}_{self.current_step}'
            
            # 创建任务对象，使用传入的参数或默认值
            task = {
                'id': task_id,
                'cpu_req': task_params.get('cpu_req', 10.0),  # CPU需求
                'memory_req': task_params.get('memory_req', 10.0),  # 内存需求
                'priority': task_params.get('priority', 3),  # 优先级
                'duration': task_params.get('duration', 10),  # 持续时间
                'type': task_params.get('type', 'compute')  # 任务类型
            }
            
            self.pending_tasks.append(task)
            env_logger.debug(f"手动添加任务: {task['id']} - CPU需求:{task['cpu_req']:.1f}, 内存需求:{task['memory_req']:.1f}, 优先级:{task['priority']}")
            return True
        else:
            env_logger.warning("任务队列已满，无法添加新任务")
            return False
    
    def _generate_tasks(self):
        """生成新任务"""
        import random
        
        for _ in range(self.task_generation_rate):
            if len(self.pending_tasks) < self.max_pending_tasks:
                # 随机选择任务类型
                task_types = ['compute', 'storage', 'network']
                task_type = random.choice(task_types)
                
                task = {
                    'id': f'task_{self.current_step}_{len(self.pending_tasks)}',
                    'cpu_req': random.uniform(5, 30),  # CPU需求
                    'memory_req': random.uniform(5, 25),  # 内存需求
                    'priority': random.randint(1, 5),  # 优先级
                    'type': task_type,  # 任务类型
                    'duration': random.randint(5, 15)  # 持续时间
                }
                self.pending_tasks.append(task)
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取每个智能体的观测"""
        observations = {}
        
        for i in range(self.num_servers):
            # 本地服务器状态
            local_state = np.array([
                self.server_cpu_usage[i] / self.server_cpu_capacity,  # CPU使用率
                self.server_memory_usage[i] / self.server_memory_capacity,  # 内存使用率
                len(self.server_current_tasks[i]) / self.server_max_tasks  # 任务数量比例
            ])
            
            # 邻居服务器状态（这里简化为所有其他服务器的平均状态）
            neighbor_states = []
            for j in range(self.num_servers):
                if i != j:
                    neighbor_states.extend([
                        self.server_cpu_usage[j] / self.server_cpu_capacity,
                        self.server_memory_usage[j] / self.server_memory_capacity
                    ])
            
            # 待处理任务信息（最多3个任务的信息）
            task_info = []
            for task in self.pending_tasks[:3]:  # 最多考虑前3个任务
                # 将任务类型转换为数值编码
                task_type = task.get('type', 'compute')
                if task_type == 'compute':
                    type_encoded = 0.0
                elif task_type == 'storage':
                    type_encoded = 0.5
                elif task_type == 'network':
                    type_encoded = 1.0
                else:
                    type_encoded = 0.0  # 默认为compute
                
                task_info.extend([
                    task['cpu_req'] / self.server_cpu_capacity,
                    task['memory_req'] / self.server_memory_capacity,
                    task['priority'] / 5.0,  # 标准化优先级
                    task['duration'] / 20.0,  # 标准化持续时间
                    type_encoded  # 编码后的任务类型
                ])
            
            # 填充任务信息到固定长度
            while len(task_info) < 3 * 5:  # 3个任务 * 5个特征
                task_info.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            # 添加全局负载均衡指标
            # 计算所有服务器的平均资源利用率
            total_cpu_usage = sum(self.server_cpu_usage) / (self.num_servers * self.server_cpu_capacity)
            total_mem_usage = sum(self.server_memory_usage) / (self.num_servers * self.server_memory_capacity)
            total_tasks = sum(len(tasks) for tasks in self.server_current_tasks) / (self.num_servers * self.server_max_tasks)
            
            # 计算资源利用率的标准差（负载均衡指标）
            cpu_utils = [usage / self.server_cpu_capacity for usage in self.server_cpu_usage]
            mem_utils = [usage / self.server_memory_capacity for usage in self.server_memory_usage]
            cpu_std = np.std(cpu_utils)
            mem_std = np.std(mem_utils)
            
            # 添加负载均衡相关特征
            load_balance_features = np.array([total_cpu_usage, total_mem_usage, total_tasks, cpu_std, mem_std])
            
            # 组合所有观测
            obs = np.concatenate([local_state, neighbor_states, task_info, load_balance_features])
            observations[f'server_{i}'] = obs.astype(np.float32)
        
        return observations
    
    def render(self, mode='human'):
        """渲染环境状态"""
        env_logger.debug(f"Step: {self.current_step}/{self.max_steps}")
        env_logger.debug(f"Pending tasks: {len(self.pending_tasks)}")
        env_logger.debug(f"Completed tasks: {self.completed_tasks}")
        env_logger.debug(f"Dropped tasks: {self.dropped_tasks}")
        env_logger.debug("Server status:")
        for i in range(self.num_servers):
            env_logger.debug(f"  Server {i}: CPU={self.server_cpu_usage[i]:.1f}%, "
                  f"Memory={self.server_memory_usage[i]:.1f}%, "
                  f"Tasks={len(self.server_current_tasks[i])}")