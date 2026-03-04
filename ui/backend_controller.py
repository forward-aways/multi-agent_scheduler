"""
多智能体任务调度系统UI后端集成模块
连接UI与多智能体调度后端
"""

import sys
import os
import threading
import time
import random
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.multi_agent_server_env import MultiAgentServerEnv
from environments.multi_agent_drone_env import MultiAgentDroneEnv
from environments.multi_agent_logistics_env import MultiAgentLogisticsEnv
from train.server.train_maddpg_server import MADDPGTrainer, MADDPGAgent
from train.server.train_mappo_server import MAPPOTrainer, MAPPOAgent
from train.drone.train_mappo_encirclement import DroneMAPPOAgent
from train.logistics.train_mappo_logistics import MAPPOTrainer as LogisticsMAPPOTrainer, LogisticsMAPPOAgent
from utils.logging_config import backend_logger


class BackendController(QObject):
    """后端控制器，负责与多智能体系统交互"""
    
    # 定义信号，用于通知UI更新
    server_data_updated = pyqtSignal(object)  # 服务器数据更新
    drone_data_updated = pyqtSignal(object)   # 无人机数据更新
    logistics_data_updated = pyqtSignal(object)  # 物流数据更新
    training_status_updated = pyqtSignal(str, float)  # 训练状态更新
    inference_status_updated = pyqtSignal(str)  # 推理状态更新
    system_error = pyqtSignal(str)  # 系统错误
    
    def __init__(self):
        super().__init__()
        self.env = None
        self.drone_env = None  # 无人机环境
        self.drone_agents = []  # 无人机智能体
        self.logistics_env = None  # 物流环境
        self.trainer = None
        self.is_running = False
        self.is_training = False
        self.is_inferring = False
        self.current_mode = 'server'  # 当前模式，默认为服务器调度
        self.pending_user_tasks = []  # 存储用户添加的任务，等待环境初始化后添加
        self.current_algorithm = 'mappo'  # 当前使用的算法，默认为MAPPO
        self.custom_drone_positions = None  # 用户自定义的无人机位置信息
        self.current_drone_task_type = 'formation'  # 当前选择的无人机任务类型（默认为队形）
        self.current_drone_formation_type = 'triangle'  # 当前选择的无人机队形类型
        
    def initialize_environment(self, config=None):
        """初始化环境"""
        try:
            if config is None:
                config = {
                    'num_servers': 5,
                    'server_cpu_capacity': 100.0,
                    'server_memory_capacity': 100.0,
                    'server_max_tasks': 10,
                    'task_generation_rate': 5,
                    'max_pending_tasks': 50,
                    'max_steps': 100
                }
            
            # 根据保存的模式设置或默认设置手动任务模式
            if hasattr(self, 'pending_manual_auto_mode'):
                # 如果有之前保存的模式设置，使用它
                config['manual_task_mode'] = (self.pending_manual_auto_mode == 'manual')
                # 清除保存的模式设置
                delattr(self, 'pending_manual_auto_mode')
            else:
                # 默认使用手动模式
                config['manual_task_mode'] = True
            
            self.env = MultiAgentServerEnv(config)
            
            # 处理暂存的用户任务
            for task_params in self.pending_user_tasks:
                if hasattr(self.env, 'add_task'):
                    success = self.env.add_task(task_params)
                    if success:
                        backend_logger.info(f"暂存任务已添加到环境: {task_params}")
                    else:
                        backend_logger.warning(f"暂存任务添加失败: {task_params}")
            
            # 清空暂存的任务列表
            self.pending_user_tasks = []
            
            return True
        except Exception as e:
            self.system_error.emit(f"初始化环境失败: {str(e)}")
            return False
    
    def set_algorithm(self, algorithm: str):
        """设置使用的算法"""
        if algorithm in ['maddpg', 'mappo']:
            self.current_algorithm = algorithm
            backend_logger.info(f"算法已设置为: {algorithm}")
            
            # 如果环境已初始化，重新加载对应的模型
            if self.env is not None:
                backend_logger.info(f"重新加载 {algorithm} 模型")
                self.load_trained_models()
            
            return True
        else:
            self.system_error.emit(f"不支持的算法: {algorithm}")
            return False
    
    def load_trained_models(self):
        """加载训练好的模型"""
        try:
            import torch
            import os
            
            if self.env is None:
                if not self.initialize_environment():
                    return False
            
            # 根据当前算法加载模型
            if self.current_algorithm == 'maddpg':
                # 加载MADDPG模型
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "multi_agent_server", "maddpg")
                
                algo_config = {
                    'actor_lr': 3e-4,
                    'critic_lr': 1e-3,
                    'gamma': 0.99,
                    'tau': 0.005,
                    'epsilon': 1.0,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.01,
                    'entropy_coef': 0.01,
                    'grad_clip': 0.5,
                    'buffer_size': 100000,
                    'batch_size': 128
                }
                
                self.trainer = MADDPGTrainer(self.env, algo_config)
                
                # 加载每个服务器的模型
                for i in range(self.env.num_servers):
                    model_path = os.path.join(model_dir, f"server_{i}_agent.pth")
                    if os.path.exists(model_path):
                        self.trainer.agents[i].load_model(model_path)
                        backend_logger.info(f"已加载MADDPG模型: {model_path}")
                    else:
                        backend_logger.warning(f"警告: 模型文件不存在: {model_path}")
            
            elif self.current_algorithm == 'mappo':
                # 加载MAPPO模型
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "multi_agent_server", "mappo")
                
                algo_config = {
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
                
                self.trainer = MAPPOTrainer(self.env, algo_config)
                
                # 加载每个服务器的模型
                for i in range(self.env.num_servers):
                    model_path = os.path.join(model_dir, f"server_{i}_agent.pth")
                    if os.path.exists(model_path):
                        self.trainer.agents[i].load_model(model_path)
                        backend_logger.info(f"已加载MAPPO模型: {model_path}")
                    else:
                        backend_logger.warning(f"警告: 模型文件不存在: {model_path}")
            
            return True
            
        except Exception as e:
            self.system_error.emit(f"加载模型失败: {str(e)}")
            return False
    
    def load_logistics_models(self):
        """加载物流调度训练好的模型"""
        try:
            import torch
            import os
            
            backend_logger.info("开始加载物流模型...")
            
            if self.logistics_env is None:
                backend_logger.error("物流环境未初始化")
                return False
            
            # 加载MAPPO模型
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "multi_agent_logistics", "mappo", "best")
            backend_logger.info(f"模型目录: {model_dir}")
            
            algo_config = {
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
            
            backend_logger.info("初始化LogisticsMAPPOTrainer...")
            self.logistics_trainer = LogisticsMAPPOTrainer(self.logistics_env, algo_config)
            backend_logger.info(f"LogisticsMAPPOTrainer初始化完成，智能体数量: {len(self.logistics_trainer.agents)}")
            
            # 加载仓库智能体模型
            for i in range(self.logistics_env.num_warehouses):
                model_path = os.path.join(model_dir, f"warehouse_{i}_agent.pth")
                backend_logger.info(f"尝试加载仓库模型: {model_path}")
                if os.path.exists(model_path):
                    self.logistics_trainer.agents[f'warehouse_{i}'].load(model_path)
                    backend_logger.info(f"已加载物流仓库模型: {model_path}")
                else:
                    backend_logger.warning(f"警告: 物流仓库模型文件不存在: {model_path}")
            
            # 加载车辆智能体模型
            for i in range(self.logistics_env.num_vehicles):
                model_path = os.path.join(model_dir, f"vehicle_{i}_agent.pth")
                backend_logger.info(f"尝试加载车辆模型: {model_path}")
                if os.path.exists(model_path):
                    self.logistics_trainer.agents[f'vehicle_{i}'].load(model_path)
                    backend_logger.info(f"已加载物流车辆模型: {model_path}")
                else:
                    backend_logger.warning(f"警告: 物流车辆模型文件不存在: {model_path}")
            
            backend_logger.info("物流模型加载完成")
            return True
            
        except Exception as e:
            import traceback
            backend_logger.error(f"加载物流模型失败：{str(e)}")
            backend_logger.error(f"错误堆栈：{traceback.format_exc()}")
            self.system_error.emit(f"加载物流模型失败：{str(e)}")
            return False
    
    def load_drone_models(self):
        """加载无人机训练好的模型"""
        try:
            import torch
            import os
            
            backend_logger.info("开始加载无人机模型...")
            
            if self.drone_env is None:
                backend_logger.error("无人机环境未初始化")
                return False
            
            # 根据任务类型确定模型目录
            if self.current_drone_task_type == 'formation':
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "models", "multi_agent_drone", "mappo", "formation", 
                                        self.current_drone_formation_type)
            elif self.current_drone_task_type == 'encirclement':
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "models", "multi_agent_drone", "mappo", "encirclement")
            else:
                backend_logger.warning(f"不支持的无人机任务类型：{self.current_drone_task_type}")
                return False
            
            backend_logger.info(f"模型目录：{model_dir}")
            
            # 根据任务类型导入正确的训练器类
            if self.current_drone_task_type == 'encirclement':
                # 包围任务使用离散动作版本
                from train.drone.train_mappo_encirclement import MAPPOTrainer as DroneMAPPOTrainer
                algo_config = {
                    'actor_lr': 3e-4,
                    'critic_lr': 1e-3,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_epsilon': 0.2,
                    'entropy_coef': 0.05,
                    'value_coef': 0.5,
                    'ppo_epochs': 10,
                    'mini_batch_size': 64
                }
            else:
                # 队形任务也使用离散动作版本（与包围任务一致）
                from train.drone.train_mappo_formation import MAPPOTrainer as DroneMAPPOTrainer
                algo_config = {
                    'actor_lr': 3e-4,
                    'critic_lr': 1e-3,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_epsilon': 0.2,
                    'entropy_coef': 0.05,
                    'value_coef': 0.5,
                    'ppo_epochs': 10,
                    'mini_batch_size': 64
                }
            
            backend_logger.info(f"初始化 DroneMAPPOTrainer (任务类型: {self.current_drone_task_type})...")
            self.drone_trainer = DroneMAPPOTrainer(self.drone_env, algo_config)
            backend_logger.info(f"DroneMAPPOTrainer 初始化完成，智能体数量：{len(self.drone_trainer.agents)}")
            
            # 加载每个无人机的模型
            for i in range(self.drone_env.num_drones):
                model_path = os.path.join(model_dir, f"drone_{i}_agent.pth")
                backend_logger.info(f"尝试加载无人机模型：{model_path}")
                if os.path.exists(model_path):
                    self.drone_trainer.agents[f'drone_{i}'].load(model_path)
                    backend_logger.info(f"已加载无人机模型：{model_path}")
                else:
                    backend_logger.warning(f"警告：无人机模型文件不存在：{model_path}")
            
            # 将 trainer 的智能体列表赋值给 drone_agents，供 _run_drone_mission 使用
            self.drone_agents = [self.drone_trainer.agents[f'drone_{i}'] for i in range(self.drone_env.num_drones)]
            backend_logger.info(f"无人机模型加载完成，共 {len(self.drone_agents)} 个智能体")
            return True
            
        except Exception as e:
            import traceback
            backend_logger.error(f"加载无人机模型失败：{str(e)}")
            backend_logger.error(f"错误堆栈：{traceback.format_exc()}")
            self.system_error.emit(f"加载无人机模型失败：{str(e)}")
            return False
    
    def start_training(self, episodes=500):
        """开始训练"""
        if self.is_training:
            self.system_error.emit("训练已在进行中")
            return
        
        if self.env is None:
            if not self.initialize_environment():
                return
        
        self.is_training = True
        
        # 在单独的线程中运行训练
        training_thread = threading.Thread(target=self._run_training, args=(episodes,))
        training_thread.daemon = True
        training_thread.start()
    
    def _run_training(self, episodes):
        """运行训练循环"""
        try:
            import torch
            # 初始化训练器
            algo_config = {
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'gamma': 0.95,
                'tau': 0.01,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'buffer_size': 10000,
                'batch_size': 64
            }
            
            self.trainer = MADDPGTrainer(self.env, algo_config)
            
            backend_logger.info(f"开始训练多智能体系统，共 {episodes} 回合")
            
            episode_rewards = []
            
            for episode in range(episodes):
                if not self.is_training:
                    break
                
                obs, _ = self.env.reset()
                total_reward = 0
                step_count = 0
                
                # 转换观测为tensor
                states = {f'server_{i}': torch.FloatTensor(obs[f'server_{i}']).unsqueeze(0) 
                         for i in range(self.trainer.num_agents)}
                
                while step_count < self.env.max_steps:
                    if not self.is_training:
                        break
                    
                    # 所有智能体选择动作（添加噪声用于训练）
                actions = {}
                for i, agent in enumerate(self.trainer.agents):
                    state_tensor = states[f'server_{i}']
                    action = agent.select_action(state_tensor, add_noise=True)
                    actions[f'server_{i}'] = action
                    
                    # 执行动作
                    next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                    
                    # 转换为tensor
                    next_states = {f'server_{i}': torch.FloatTensor(next_obs[f'server_{i}']).unsqueeze(0) 
                                  for i in range(self.trainer.num_agents)}
                    
                    # 存储经验
                    for i, agent in enumerate(self.trainer.agents):
                        state = states[f'server_{i}'].squeeze(0).numpy()
                        action = actions[f'server_{i}']
                        reward = rewards[f'server_{i}']
                        next_state = next_states[f'server_{i}'].squeeze(0).numpy()
                        done = terminated[f'server_{i}'] or truncated[f'server_{i}']
                        
                        agent.add_experience(state, action, reward, next_state, done)
                    
                    # 检查是否结束
                    if all(terminated.values()) or all(truncated.values()):
                        break
                    
                    # 更新状态
                    states = next_states
                    total_reward += sum(rewards.values())
                    step_count += 1
                
                # 在每个回合结束后训练
                if len(self.trainer.agents[0].replay_buffer) >= self.trainer.agents[0].batch_size:
                    # 从经验回放缓冲区采样
                    indices = random.sample(range(len(self.trainer.agents[0].replay_buffer)), 
                                          self.trainer.agents[0].batch_size)
                    batch = [self.trainer.agents[0].replay_buffer[i] for i in indices]
                    
                    # 解包批次数据
                    batch_states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
                    batch_actions = torch.FloatTensor(np.array([exp[1] for exp in batch]))
                    batch_rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).unsqueeze(1)
                    batch_next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
                    batch_dones = torch.BoolTensor(np.array([exp[4] for exp in batch])).unsqueeze(1)
                    
                    # 对每个智能体分别训练
                    for i, agent in enumerate(self.trainer.agents):
                        agent.train_on_batch(
                            batch_states, batch_actions, batch_rewards, 
                            batch_next_states, batch_dones, self.trainer.agents
                        )
                
                episode_rewards.append(total_reward)
                
                # 发送训练进度更新
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                self.training_status_updated.emit(f"训练进度: {episode + 1}/{episodes}", avg_reward)
                
                if (episode + 1) % 10 == 0:
                    backend_logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
            
            backend_logger.info("训练完成!")
            self.is_training = False
            self.training_status_updated.emit("训练完成!", avg_reward)
            
        except Exception as e:
            self.system_error.emit(f"训练过程中出错: {str(e)}")
            self.is_training = False
    
    def start_inference(self):
        """开始推理"""
        backend_logger.info("开始推理")
        if self.is_inferring:
            self.system_error.emit("推理已在进行中")
            backend_logger.warning("推理已在进行中")
            return
        
        # 检查当前模式和任务状态
        is_manual_mode = hasattr(self, '_current_manual_auto_mode') and self._current_manual_auto_mode == 'manual'
        backend_logger.debug(f"当前是否为手动模式: {is_manual_mode}")
        
        # 强制检查：在手动模式下，如果没有任务则不允许启动推理
        if is_manual_mode:
            # 检查暂存的用户任务
            has_pending_user_tasks = hasattr(self, 'pending_user_tasks') and len(self.pending_user_tasks) > 0
            backend_logger.debug(f"暂存的用户任务数量: {len(self.pending_user_tasks) if hasattr(self, 'pending_user_tasks') else 0}")
            
            # 检查环境中的任务（不管环境是否已初始化）
            has_env_tasks = False
            if self.env is not None:
                # 检查待处理任务
                if hasattr(self.env, 'pending_tasks'):
                    pending_count = len(self.env.pending_tasks)
                    backend_logger.debug(f"环境中的待处理任务数量: {pending_count}")
                    if pending_count > 0:
                        has_env_tasks = True
                # 检查运行中的任务
                if hasattr(self.env, 'server_current_tasks'):
                    running_count = sum(len(tasks) for tasks in self.env.server_current_tasks)
                    backend_logger.debug(f"环境中的运行任务总数: {running_count}")
                    if running_count > 0:
                        has_env_tasks = True
            
            # 检查物流环境的订单
            if self.logistics_env is not None:
                if hasattr(self.logistics_env, 'pending_orders'):
                    order_count = len(self.logistics_env.pending_orders)
                    backend_logger.debug(f"物流环境中的待处理订单数量: {order_count}")
                    if order_count > 0:
                        has_env_tasks = True
            
            # 检查暂存的物流订单
            if hasattr(self, 'pending_logistics_orders') and len(self.pending_logistics_orders) > 0:
                backend_logger.debug(f"暂存的物流订单数量: {len(self.pending_logistics_orders)}")
                has_env_tasks = True
            
            backend_logger.debug(f"has_pending_user_tasks: {has_pending_user_tasks}, has_env_tasks: {has_env_tasks}")
            
            # 如果在手动模式下没有任务，则不允许启动推理
            if not has_pending_user_tasks and not has_env_tasks:
                backend_logger.info("手动模式下没有任务，显示提示并返回")
                self.system_error.emit("手动模式下没有任务，请先添加任务再启动调度")
                return
            else:
                backend_logger.info("手动模式下找到任务，继续执行")
        
        try:
            import torch
            # 根据当前模式初始化相应的环境
            if self.current_mode == 'server':
                if self.env is None:
                    config = {
                        'num_servers': 5,
                        'server_cpu_capacity': 100.0,
                        'server_memory_capacity': 100.0,
                        'server_max_tasks': 10,
                        'task_generation_rate': 5,
                        'max_pending_tasks': 50,
                        'max_steps': 100
                    }
                    
                    # 设置手动/自动模式
                    config['manual_task_mode'] = (getattr(self, '_current_manual_auto_mode', 'auto') == 'manual')
                    
                    # 初始化环境
                    self.env = MultiAgentServerEnv(config)
                    
                    # 加载预训练模型（根据当前算法）
                    if not self.load_trained_models():
                        self.system_error.emit("加载模型失败")
                        return
            
            # 对于无人机和物流模式，我们可以扩展以使用适当的环境
            elif self.current_mode == 'drone':
                # 初始化无人机环境
                backend_logger.info("无人机模式推理开始...")
                if self.drone_env is None:
                    # 根据任务类型配置环境
                    if self.current_drone_task_type == 'formation':
                        config = {
                            'num_drones': 3,
                            'max_speed': 2.0,  # 与训练保持一致
                            'battery_capacity': 100.0,
                            'payload_capacity': 5.0,
                            'space_size': [100, 100, 50],
                            'task_type': 'formation',
                            'formation_type': self.current_drone_formation_type,
                            'max_steps': 200
                        }
                    elif self.current_drone_task_type == 'encirclement':
                        config = {
                            'num_drones': 3,
                            'max_speed': 2.0,  # 与训练保持一致
                            'battery_capacity': 100.0,
                            'payload_capacity': 5.0,
                            'space_size': [100, 100, 50],
                            'task_type': 'encirclement',
                            'max_steps': 200
                        }
                    else:  # inspection
                        config = {
                            'num_drones': 3,
                            'max_speed': 10.0,
                            'battery_capacity': 100.0,
                            'payload_capacity': 5.0,
                            'space_size': [100, 100, 50],
                            'task_type': 'inspection',
                            'num_waypoints': 4,
                            'max_steps': 200
                        }
                    
                    self.drone_env = MultiAgentDroneEnv(config)
                    backend_logger.info(f"无人机环境初始化完成，任务类型：{self.current_drone_task_type}")
                    
                    # 加载预训练模型
                    if not self.load_drone_models():
                        self.system_error.emit("加载无人机模型失败")
                        return
            elif self.current_mode == 'logistics':
                # 初始化物流环境
                backend_logger.info("物流模式推理开始...")
                if self.logistics_env is None:
                    # 检查是否为手动模式
                    is_manual_mode = getattr(self, '_current_manual_auto_mode', 'auto') == 'manual'
                    backend_logger.info(f"物流环境初始化，模式: {'手动' if is_manual_mode else '自动'}")
                    
                    config = {
                        'num_warehouses': 3,
                        'num_vehicles': 5,
                        'warehouse_capacity': 100,
                        'vehicle_capacity': 20,
                        'vehicle_speed': 5.0,
                        'order_generation_rate': 2,
                        'max_pending_orders': 15,
                        'map_size': [100.0, 100.0],
                        'max_steps': 200,
                        'manual_mode': is_manual_mode  # 传递手动模式标志
                    }
                    self.logistics_env = MultiAgentLogisticsEnv(config)
                    
                    # 加载预训练模型
                    if not self.load_logistics_models():
                        self.system_error.emit("加载物流模型失败")
                        return
                
                # 在环境初始化后，将暂存的订单添加到环境中
                if hasattr(self, 'pending_logistics_orders') and len(self.pending_logistics_orders) > 0:
                    backend_logger.debug(f"发现 {len(self.pending_logistics_orders)} 个暂存订单，添加到环境中")
                    for order in self.pending_logistics_orders:
                        self._add_logistics_order(order)
                    # 清空暂存的订单列表
                    self.pending_logistics_orders.clear()
                    backend_logger.debug(f"暂存订单已添加到环境并清空暂存列表")
            
            # 在环境初始化后，将暂存的任务添加到环境中
            if hasattr(self, 'pending_user_tasks') and len(self.pending_user_tasks) > 0:
                backend_logger.debug(f"发现 {len(self.pending_user_tasks)} 个暂存任务，添加到环境中")
                for task in self.pending_user_tasks:
                    self.env.add_task(task)  # 添加任务到环境
                # 清空暂存的任务列表
                self.pending_user_tasks.clear()
                backend_logger.debug(f"暂存任务已添加到环境并清空暂存列表")
            
            # 在环境初始化后，再次进行强制检查（双重保险）
            if (hasattr(self, '_current_manual_auto_mode') and 
                self._current_manual_auto_mode == 'manual'):
                
                backend_logger.debug("环境初始化后再次检查")
                # 重新检查环境中的任务
                has_env_tasks = False
                if hasattr(self.env, 'pending_tasks'):
                    pending_count = len(self.env.pending_tasks)
                    backend_logger.debug(f"环境初始化后待处理任务数量: {pending_count}")
                    if pending_count > 0:
                        has_env_tasks = True
                if hasattr(self.env, 'server_current_tasks'):
                    running_count = sum(len(tasks) for tasks in self.env.server_current_tasks)
                    backend_logger.debug(f"环境初始化后运行任务总数: {running_count}")
                    if running_count > 0:
                        has_env_tasks = True
                
                # 检查物流环境的订单
                if self.logistics_env is not None:
                    if hasattr(self.logistics_env, 'pending_orders'):
                        order_count = len(self.logistics_env.pending_orders)
                        backend_logger.debug(f"环境初始化后物流订单数量: {order_count}")
                        if order_count > 0:
                            has_env_tasks = True
                
                # 如果在手动模式下仍然没有任务，则不允许启动推理
                if not has_env_tasks:
                    backend_logger.info("环境初始化后手动模式下没有任务，显示提示并返回")
                    self.system_error.emit("手动模式下没有任务，请先添加任务再启动调度")
                    return
                else:
                    backend_logger.info("环境初始化后手动模式下找到任务，继续执行")
        
        except Exception as e:
            self.system_error.emit(f"初始化环境失败: {str(e)}")
            return
        
        self.is_inferring = True
        
        # 在单独的线程中运行推理
        inference_thread = threading.Thread(target=self._run_inference)
        inference_thread.daemon = True
        inference_thread.start()

    def _run_inference(self):
        """运行推理循环"""
        backend_logger.debug("_run_inference 开始执行")
        try:
            import torch
            
            # 根据当前模式选择环境
            if self.current_mode == 'logistics':
                return self._run_logistics_inference()
            
            # 检查当前模式和任务状态
            is_manual_mode = getattr(self, '_current_manual_auto_mode', 'auto') == 'manual'
            backend_logger.debug(f"推理循环 - 是否手动模式: {is_manual_mode}")
            
            # 在手动模式下，如果环境有暂存任务，需要先保存这些任务
            saved_tasks = []
            if is_manual_mode and hasattr(self.env, 'pending_tasks'):
                # 保存当前环境中的任务，因为reset会清空它们
                saved_tasks = self.env.pending_tasks.copy()
                backend_logger.debug(f"保存了 {len(saved_tasks)} 个任务，准备在重置后重新添加")
            
            # 重置环境
            backend_logger.debug("重置环境")
            obs, _ = self.env.reset()
            
            # 如果是手动模式且有保存的任务，重新添加它们
            if is_manual_mode and saved_tasks:
                backend_logger.debug(f"保存了 {len(saved_tasks)} 个任务，准备在重置后重新添加")
                for task in saved_tasks:
                    self.env.add_task(task)
            
            total_reward = 0
            step_count = 0
            
            # 转换观测为tensor
            states = {f'server_{i}': torch.FloatTensor(obs[f'server_{i}']).unsqueeze(0) 
                     for i in range(self.trainer.num_agents)}
            
            # 检查环境中的任务状态
            if hasattr(self.env, 'pending_tasks'):
                backend_logger.debug(f"推理循环 - 待处理任务数量: {len(self.env.pending_tasks)}")
            if hasattr(self.env, 'server_current_tasks'):
                running_count = sum(len(tasks) for tasks in self.env.server_current_tasks)
                backend_logger.debug(f"推理循环 - 运行任务总数: {running_count}")
            
            while step_count < self.env.max_steps and self.is_inferring:
                backend_logger.debug(f"推理循环第 {step_count} 步")
                
                # 检查是否在手动模式下且没有任务
                if is_manual_mode:
                    has_pending_tasks = hasattr(self.env, 'pending_tasks') and len(self.env.pending_tasks) > 0
                    has_running_tasks = hasattr(self.env, 'server_current_tasks') and any(len(tasks) > 0 for tasks in self.env.server_current_tasks)
                    
                    if not has_pending_tasks and not has_running_tasks:
                        backend_logger.info("手动模式下没有任务，正常结束推理循环")
                        break
                
                # 所有智能体选择动作（推理时不加噪声，使用最佳策略）
                actions = {}
                for i in range(self.trainer.num_agents):
                    state_tensor = states[f'server_{i}']
                    
                    # 根据trainer的实际类型选择动作，而不是根据current_algorithm
                    # 因为切换算法时可能还没有重新加载模型
                    trainer_type = type(self.trainer).__name__
                    
                    if trainer_type == 'MADDPGTrainer':
                        # MADDPG推理时不需要噪声，直接使用最佳策略
                        action = self.trainer.agents[i].select_action(state_tensor, add_noise=False)
                    elif trainer_type == 'MAPPOTrainer':
                        # MAPPO推理时也不需要噪声
                        action, _, _ = self.trainer.agents[i].select_action(state_tensor, training=False)
                    else:
                        # 默认使用MADDPG方式
                        action = self.trainer.agents[i].select_action(state_tensor, add_noise=False)
                    
                    actions[f'server_{i}'] = action
                
                # 执行动作
                backend_logger.debug("执行动作")
                obs, rewards, terminated, truncated, _ = self.env.step(actions)
                
                # 转换为tensor
                next_states = {f'server_{i}': torch.FloatTensor(obs[f'server_{i}']).unsqueeze(0) 
                              for i in range(self.trainer.num_agents)}
                
                # 更新状态
                states = next_states
                total_reward += sum(rewards.values())
                step_count += 1
                
                # 更新UI显示
                backend_logger.debug("更新UI显示")
                self._update_ui_data()
                
                if all(terminated.values()) or all(truncated.values()):
                    backend_logger.debug("环境终止，退出循环")
                    break
                
                time.sleep(0.1)  # 短暂延迟以允许UI更新
            
            backend_logger.info(f"推理完成! 总奖励: {total_reward:.2f}")
            self.inference_status_updated.emit(f"推理完成! 总奖励: {total_reward:.2f}")
            self.is_inferring = False
            
        except RuntimeError as e:
            # 如果是对象删除错误，静默退出（UI组件已被销毁）
            if "wrapped C/C++ object" in str(e):
                self.is_inferring = False
                return
            else:
                backend_logger.debug(f"推理循环中RuntimeError: {str(e)}")
                self.system_error.emit(f"推理过程中出错: {str(e)}")
                self.is_inferring = False
        except Exception as e:
            backend_logger.error(f"推理循环中Exception: {str(e)}")
            self.system_error.emit(f"推理过程中出错: {str(e)}")
            self.is_inferring = False
    
    def _run_logistics_inference(self):
        """运行物流调度推理循环"""
        backend_logger.debug("_run_logistics_inference 开始执行")
        try:
            import torch
            
            # 不重置环境，直接使用当前环境状态
            backend_logger.debug("获取当前环境观测（不重置环境）")
            try:
                obs = self.logistics_env._get_observations()
                backend_logger.debug(f"获取观测成功，观测keys: {obs.keys()}")
            except Exception as e:
                backend_logger.error(f"获取观测时出错: {str(e)}")
                import traceback
                backend_logger.error(f"错误堆栈: {traceback.format_exc()}")
                raise
            
            total_reward = 0
            step_count = 0
            
            # 转换观测为tensor
            states = {}
            try:
                for agent_id in obs.keys():
                    backend_logger.debug(f"转换观测 {agent_id}: shape={obs[agent_id].shape}, dtype={obs[agent_id].dtype}")
                    states[agent_id] = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
            except Exception as e:
                backend_logger.error(f"转换观测时出错: {str(e)}")
                import traceback
                backend_logger.error(f"错误堆栈: {traceback.format_exc()}")
                raise
            
            # 检查环境中的订单状态
            backend_logger.debug(f"推理循环 - 待处理订单数量: {len(self.logistics_env.pending_orders)}")
            
            while step_count < self.logistics_env.max_steps and self.is_inferring:
                backend_logger.debug(f"推理循环第 {step_count} 步")
                
                # 所有智能体选择动作（推理时不加噪声，使用最佳策略）
                actions = {}
                for agent_id in obs.keys():
                    state_tensor = states[agent_id]
                    
                    # MAPPO推理时不需要噪声
                    try:
                        action = self.logistics_trainer.agents[agent_id].select_action(state_tensor, training=False)
                        actions[agent_id] = action
                    except Exception as e:
                        backend_logger.error(f"智能体{agent_id}选择动作时出错: {str(e)}")
                        raise
                
                # 执行动作
                backend_logger.debug("执行动作")
                try:
                    obs, rewards, terminated, truncated, _ = self.logistics_env.step(actions)
                except Exception as e:
                    backend_logger.error(f"执行动作时出错: {str(e)}")
                    backend_logger.error(f"当前actions: {actions}")
                    backend_logger.error(f"当前pending_orders: {self.logistics_env.pending_orders}")
                    backend_logger.error(f"当前warehouse_orders: {self.logistics_env.warehouse_orders}")
                    raise
                
                # 转换为tensor
                next_states = {}
                for agent_id in obs.keys():
                    next_states[agent_id] = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
                
                # 更新状态
                states = next_states
                total_reward += sum(rewards.values())
                step_count += 1
                
                # 更新UI显示
                backend_logger.debug("更新UI显示")
                self._update_logistics_ui_data()
                
                if all(terminated.values()) or all(truncated.values()):
                    backend_logger.debug("环境终止，退出循环")
                    break
                
                time.sleep(0.1)  # 短暂延迟以允许UI更新
            
            backend_logger.info(f"物流推理完成! 总奖励: {total_reward:.2f}")
            self.inference_status_updated.emit(f"物流推理完成! 总奖励: {total_reward:.2f}")
            self.is_inferring = False
            
        except RuntimeError as e:
            # 如果是对象删除错误，静默退出（UI组件已被销毁）
            if "wrapped C/C++ object" in str(e):
                self.is_inferring = False
                return
            else:
                backend_logger.debug(f"物流推理循环中RuntimeError: {str(e)}")
                self.system_error.emit(f"物流推理过程中出错: {str(e)}")
                self.is_inferring = False
        except Exception as e:
            import traceback
            error_msg = f"物流推理循环中Exception: {str(e)}"
            stack_trace = traceback.format_exc()
            backend_logger.error(error_msg)
            backend_logger.error(f"错误堆栈:\n{stack_trace}")
            self.system_error.emit(f"物流推理过程中出错: {str(e)}")
            self.is_inferring = False

    def _update_ui_data(self):
        """更新UI数据"""
        # 从实际环境获取服务器数据
        server_data = []
        if self.env is not None:
            for i in range(self.env.num_servers):
                # 计算CPU和内存使用率百分比
                cpu_percent = (self.env.server_cpu_usage[i] / self.env.server_cpu_capacity) * 100
                mem_percent = (self.env.server_memory_usage[i] / self.env.server_memory_capacity) * 100
                task_count = len(self.env.server_current_tasks[i])
                
                # 根据资源使用情况确定状态
                if cpu_percent > 80 or mem_percent > 80:
                    status = 'overloaded'
                elif task_count > 0:
                    status = 'busy'
                else:
                    status = 'idle'
                
                server_data.append({
                    'id': i,
                    'cpu': cpu_percent,
                    'memory': mem_percent,
                    'tasks': task_count,
                    'status': status,
                    'current_tasks': self.env.server_current_tasks[i],  # 添加当前任务信息
                    'cpu_capacity': self.env.server_cpu_capacity,
                    'memory_capacity': self.env.server_memory_capacity,
                    'tasks_completed': getattr(self.env, 'completed_tasks', 0),
                    'tasks_failed': getattr(self.env, 'dropped_tasks', 0)
                })
            
            # 添加待处理任务信息到第一个服务器数据中，方便UI获取
            if hasattr(self.env, 'pending_tasks'):
                server_data[0]['pending_tasks'] = self.env.pending_tasks
        else:
            # 如果环境未初始化，使用默认数据
            server_data = [
                {
                    'id': i,
                    'cpu': 0,
                    'memory': 0,
                    'tasks': 0,
                    'status': 'idle'
                } for i in range(5)
            ]
        
        # 无人机数据（模拟真实数据结构）
        drone_data = []
        # 这里可以集成真实的无人机环境，但现在使用模拟数据
        for i in range(3):
            drone_data.append({
                'id': i,
                'position': [random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 50)],
                'battery': random.randint(30, 100),
                'task': random.choice(['delivery', 'returning', 'idle']),
                'status': random.choice(['flying', 'hovering', 'charging']),
                'speed': random.uniform(0, 10)
            })
        
        # 物流数据（模拟真实数据结构）
        logistics_data = []
        for i in range(3):
            logistics_data.append({
                'id': i,
                'inventory': {
                    'A': random.randint(50, 150),
                    'B': random.randint(40, 120),
                    'C': random.randint(30, 100)
                },
                'orders': random.randint(0, 10),
                'vehicles': random.randint(1, 5),
                'status': random.choice(['normal', 'restocking', 'delayed'])
            })
        
        # 安全地发射信号更新UI（捕获可能的对象删除异常）
        try:
            self.server_data_updated.emit(server_data)
        except RuntimeError:
            # 如果对象已被删除，则停止推理
            self.is_inferring = False
            return
            
        try:
            self.drone_data_updated.emit(drone_data)
        except RuntimeError:
            self.is_inferring = False
            return
            
        try:
            self.logistics_data_updated.emit(logistics_data)
        except RuntimeError:
            self.is_inferring = False
            return
    
    def _update_logistics_ui_data(self):
        """更新物流调度UI数据"""
        # 物流数据
        logistics_data = []
        
        if self.logistics_env is not None:
            # 仓库数据
            for i in range(self.logistics_env.num_warehouses):
                # warehouse_inventory是一维数组，每个元素是一个标量
                inventory_value = float(self.logistics_env.warehouse_inventory[i])
                
                logistics_data.append({
                    'type': 'warehouse',
                    'id': i,
                    'inventory': inventory_value,
                    'orders': len(self.logistics_env.warehouse_orders[i]),
                    'position': self.logistics_env.warehouse_positions[i],
                    'status': 'normal'
                })
            
            # 车辆数据
            for i in range(self.logistics_env.num_vehicles):
                logistics_data.append({
                    'type': 'vehicle',
                    'id': i,
                    'position': self.logistics_env.vehicle_positions[i].tolist(),
                    'cargo': self.logistics_env.vehicle_cargo[i],
                    'status': self.logistics_env.vehicle_status[i],
                    'target_warehouse': int(self.logistics_env.vehicle_target_warehouse[i]),
                    'target_order': self.logistics_env.vehicle_target_order_pos[i].tolist() if self.logistics_env.vehicle_target_order_pos[i] is not None else None
                })
            
            # 订单数据（发送所有订单，包括状态）
            for i, order_info in enumerate(self.logistics_env.all_orders):
                logistics_data.append({
                    'type': 'order',
                    'id': i,
                    'position': order_info['position'].tolist() if hasattr(order_info['position'], 'tolist') else order_info['position'],
                    'quantity': order_info['quantity'],
                    'priority': order_info['priority'],
                    'status': order_info['status']
                })
            
            # 添加统计信息
            logistics_data.append({
                'type': 'stats',
                'completed_orders': self.logistics_env.completed_orders,
                'failed_orders': self.logistics_env.failed_orders,
                'pending_orders': len(self.logistics_env.pending_orders)
            })
        else:
            # 如果环境未初始化，使用默认数据
            for i in range(3):
                logistics_data.append({
                    'type': 'warehouse',
                    'id': i,
                    'inventory': 50,
                    'orders': 0,
                    'position': [20 + i * 30, 20 + i * 30],
                    'status': 'normal'
                })
            
            for i in range(5):
                logistics_data.append({
                    'type': 'vehicle',
                    'id': i,
                    'position': [20 + i * 15, 20 + i * 15],
                    'cargo': 0,
                    'status': 0,
                    'target_warehouse': 0,
                    'target_order': 0
                })
        
        # 安全地发射信号更新UI
        try:
            self.logistics_data_updated.emit(logistics_data)
        except RuntimeError:
            self.is_inferring = False
            return
        
    def set_mode(self, mode):
        """设置当前运行模式"""
        self.current_mode = mode
        backend_logger.info(f"切换到 {mode} 模式")
    
    def set_manual_auto_mode(self, mode):
        """设置手动/自动调度模式"""
        # 保存当前模式设置
        backend_logger.info(f"DEBUG: set_manual_auto_mode 被调用，设置模式为: {mode} (类型: {type(mode)})")
        self._current_manual_auto_mode = mode
        backend_logger.info(f"DEBUG: _current_manual_auto_mode 现在是: {self._current_manual_auto_mode} (类型: {type(self._current_manual_auto_mode)})")
        
        if self.env is not None:
            # 设置环境的调度模式
            if mode == 'auto':
                self.env.manual_task_mode = False
                backend_logger.info("切换到自动调度模式 - 环境将自动生成任务")
            elif mode == 'manual':
                self.env.manual_task_mode = True
                backend_logger.info("切换到手动调度模式 - 环境仅处理用户添加的任务")
        else:
            backend_logger.info(f"环境未初始化，模式将在环境初始化后设置为: {mode}")
            # 保存模式设置，等待环境初始化
            self.pending_manual_auto_mode = mode
    
    def add_task_to_env(self, task_params):
        """向环境添加新任务"""
        backend_logger.info(f"add_task_to_env 被调用，当前模式: {self.current_mode}, 参数: {task_params}")
        
        # 检查当前模式
        if self.current_mode == 'logistics':
            # 物流环境：添加订单
            backend_logger.info("调用 _add_logistics_order 添加物流订单")
            self._add_logistics_order(task_params)
        elif self.env is not None:
            # 服务器环境：添加任务
            if hasattr(self.env, 'add_task'):
                # 如果环境支持直接添加任务
                success = self.env.add_task(task_params)
                if success:
                    backend_logger.info(f"任务已成功添加到环境: {task_params}")
                else:
                    backend_logger.warning(f"任务添加失败: {task_params}")
            else:
                # 否则将任务添加到待处理队列
                new_task = {
                    'id': f'USER_TASK_{len(self.env.pending_tasks) if hasattr(self.env, "pending_tasks") else 0}',
                    'cpu_req': task_params.get('cpu_req', 10.0),
                    'memory_req': task_params.get('memory_req', 10.0),
                    'priority': task_params.get('priority', 3),
                    'type': task_params.get('type', 'compute')
                }
                if hasattr(self.env, 'pending_tasks'):
                    self.env.pending_tasks.append(new_task)
                backend_logger.info(f"任务已添加到环境: {new_task}")
        else:
            # 环境未初始化，将任务暂存到待处理队列
            self.pending_user_tasks.append(task_params)
            backend_logger.debug(f"任务已暂存，等待环境初始化: {task_params}")
    
    def _add_logistics_order(self, order_params):
        """向物流环境添加订单"""
        backend_logger.info(f"_add_logistics_order 被调用，物流环境状态: {self.logistics_env is not None}")
        
        if self.logistics_env is not None:
            # 将订单参数转换为环境需要的格式
            # order_params格式: {'type': '普通订单', 'priority': '高', 'quantity': 10, 'origin': 'N0', 'destination': 'N1'}
            
            # 将中文优先级转换为数值
            priority_map = {"高": 5, "中": 3, "低": 1}
            priority_text = order_params.get('priority', '中')
            priority_value = priority_map.get(priority_text, 3)
            
            # 生成随机位置
            position = np.random.rand(2) * self.logistics_env.map_size
            quantity = order_params.get('quantity', 10)
            
            # 创建订单
            new_order = [position, quantity, priority_value]
            
            # 添加到待处理订单队列
            if len(self.logistics_env.pending_orders) < self.logistics_env.max_pending_orders:
                self.logistics_env.pending_orders.append(new_order)
                
                # 同时添加到all_orders列表，跟踪状态
                order_info = {
                    'position': position,
                    'quantity': quantity,
                    'priority': priority_value,
                    'status': 'pending'
                }
                self.logistics_env.all_orders.append(order_info)
                
                backend_logger.info(f"物流订单已添加: 位置={position}, 数量={quantity}, 优先级={priority_value}")
            else:
                backend_logger.warning("待处理订单队列已满，无法添加新订单")
        else:
            # 物流环境未初始化，将订单暂存到待处理队列
            if not hasattr(self, 'pending_logistics_orders'):
                self.pending_logistics_orders = []
            self.pending_logistics_orders.append(order_params)
            backend_logger.info(f"物流订单已暂存，等待环境初始化: {order_params}")
    
    def stop_current_operation(self):
        """停止当前操作"""
        self.is_training = False
        self.is_inferring = False
        self.is_running = False
        
    def reset_system(self):
        """重置系统"""
        self.is_training = False
        self.is_inferring = False
        self.is_running = False
        
        # 清空暂存的任务列表
        if hasattr(self, 'pending_user_tasks'):
            self.pending_user_tasks.clear()
        
        # 重新初始化环境
        if self.env is not None:
            config = {
                'num_servers': 5,
                'server_cpu_capacity': 100.0,
                'server_memory_capacity': 100.0,
                'server_max_tasks': 10,
                'task_generation_rate': 5,
                'max_pending_tasks': 50,
                'max_steps': 100
            }
            
            # 保留当前的手动/自动模式设置
            if hasattr(self, '_current_manual_auto_mode'):
                config['manual_task_mode'] = (self._current_manual_auto_mode == 'manual')
            else:
                # 默认使用手动模式
                config['manual_task_mode'] = True
                
            self.env = MultiAgentServerEnv(config)
        
        # 重置物流环境
        if self.logistics_env is not None:
            backend_logger.info("重置物流环境")
            self.logistics_env.reset()
            backend_logger.info("物流环境已重置，订单已清空")
        
        # 发送重置信号给UI组件，让它们重置显示
        try:
            self.server_data_updated.emit({})
            self.drone_data_updated.emit({})
            self.logistics_data_updated.emit({})
        except:
            pass  # 如果UI组件已销毁，忽略错误
    
    def change_drone_task_type(self, task_type):
        """切换无人机任务类型"""
        # 保存当前选择的任务类型
        self.current_drone_task_type = task_type
        
        if self.drone_env is not None:
            self.drone_env.set_task_type(task_type)
            backend_logger.info(f"无人机任务类型切换为: {task_type}")
            
            # 重新加载对应任务的模型
            if hasattr(self, 'drone_agents') and len(self.drone_agents) > 0:
                self._load_drone_model(task_type)
        else:
            backend_logger.warning("无人机环境未初始化，任务类型将在环境初始化时设置")
    
    def change_drone_formation_type(self, formation_type):
        """切换无人机队形类型"""
        # 保存当前选择的队形类型
        self.current_drone_formation_type = formation_type
        
        if self.drone_env is not None:
            self.drone_env.set_formation_type(formation_type)
            backend_logger.info(f"无人机队形类型切换为: {formation_type}")
            
            # 重新加载对应队形的模型
            if hasattr(self, 'drone_agents') and len(self.drone_agents) > 0:
                self._load_drone_model('formation', formation_type)
        else:
            backend_logger.warning("无人机环境未初始化，队形类型将在环境初始化时设置")
    
    def set_custom_drone_positions(self, positions: dict):
        """
        设置用户自定义的无人机位置信息
        
        参数:
            positions: 位置信息字典
                {
                    'task_type': 'inspection' or 'formation',
                    'start_point': [x, y, z],
                    'end_point': [x, y, z],
                    'waypoints': [[x, y, z], ...]  # 仅巡检任务
                }
        """
        self.custom_drone_positions = positions
        backend_logger.info(f"用户自定义位置已设置: {positions}")
    
    def start_drone_mission(self):
        """开始无人机任务"""
        try:
            if self.drone_env is None:
                # 根据任务类型设置max_speed（必须与训练一致）
                if self.current_drone_task_type in ['formation', 'encirclement']:
                    max_speed = 2.0  # 队形和包围任务使用2.0
                else:
                    max_speed = 10.0  # 巡检任务使用10.0
                
                config = {
                    'num_drones': 3,
                    'max_speed': max_speed,  # 与训练保持一致
                    'battery_capacity': 100.0,
                    'payload_capacity': 5.0,
                    'space_size': [100, 100, 50],
                    'task_type': self.current_drone_task_type,
                    'num_waypoints': 4,
                    'formation_type': self.current_drone_formation_type,
                    'max_steps': 200
                }
                self.drone_env = MultiAgentDroneEnv(config, custom_positions=self.custom_drone_positions)
                
                # 加载训练好的MAPPO模型，使用当前选择的任务类型和队形类型
                self._load_drone_model(self.current_drone_task_type, self.current_drone_formation_type)
            
            self.is_inferring = True
            
            # 在单独的线程中运行无人机任务
            drone_thread = threading.Thread(target=self._run_drone_mission)
            drone_thread.daemon = True
            drone_thread.start()
            
            backend_logger.info("无人机任务已启动")
            
        except Exception as e:
            self.system_error.emit(f"启动无人机任务失败: {str(e)}")
    
    def _load_drone_model(self, task_type=None, formation_type=None):
        """加载训练好的无人机模型"""
        try:
            import torch
            import os
            
            # 确定要加载的任务类型
            if task_type is None:
                task_type = self.drone_env.task_type if self.drone_env else 'formation'
            
            # 确定要加载的队形类型
            if formation_type is None:
                formation_type = self.current_drone_formation_type if task_type == 'formation' else None
            
            # 根据任务类型选择模型目录
            if task_type == 'inspection':
                backend_logger.error("巡检任务已废弃，请使用队形任务或协同包围任务")
                return False
            elif task_type == 'formation':
                if formation_type is None:
                    formation_type = 'triangle'  # 默认三角形
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "multi_agent_drone", "mappo", "formation", formation_type)
                backend_logger.info(f"加载{formation_type}队形任务模型: {model_dir}")
            elif task_type == 'encirclement':
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "multi_agent_drone", "mappo", "encirclement")
                backend_logger.info(f"加载协同包围任务模型: {model_dir}")
            else:
                backend_logger.error(f"未知的任务类型: {task_type}")
                return False
            
            # 检查模型目录是否存在
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                backend_logger.warning(f"无人机模型目录不存在，已创建: {model_dir}")
                backend_logger.warning(f"请先训练{task_type}任务模型")
                return False
            
            # 创建智能体
            self.drone_agents = []
            for i in range(self.drone_env.num_drones):
                agent_id = f"drone_{i}"
                state_dim = self.drone_env.observation_spaces[agent_id].shape[0]
                # 对于离散动作空间，使用 .n 获取动作数量
                action_space = self.drone_env.action_spaces[agent_id]
                if hasattr(action_space, 'n'):
                    action_dim = action_space.n  # 离散动作空间
                else:
                    action_dim = action_space.shape[0]  # 连续动作空间
                
                # 创建MAPPO智能体
                agent = DroneMAPPOAgent(agent_id, state_dim, action_dim, {
                    'actor_lr': 3e-4,
                    'critic_lr': 1e-3,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_epsilon': 0.2,
                    'entropy_coef': 0.01,
                    'value_coef': 0.5,
                    'ppo_epochs': 10,
                    'mini_batch_size': 64,
                    'action_std': 0.5
                })
                
                # 尝试加载模型
                model_path = os.path.join(model_dir, f"{agent_id}_agent.pth")
                if os.path.exists(model_path):
                    agent.load(model_path)
                    backend_logger.info(f"已加载{task_type}任务模型: {model_path}")
                else:
                    backend_logger.warning(f"{task_type}任务模型文件不存在: {model_path}")
                    backend_logger.warning("将使用随机动作")
                
                self.drone_agents.append(agent)
            
            return True
            
        except Exception as e:
            backend_logger.error(f"加载无人机模型失败: {str(e)}")
            return False
    
    def _run_drone_mission(self):
        """运行无人机任务循环"""
        try:
            obs, _ = self.drone_env.reset()
            total_reward = 0
            step_count = 0
            
            while step_count < self.drone_env.max_steps and self.is_inferring:
                # 使用训练好的模型选择动作
                actions = {}
                for i, agent in enumerate(self.drone_agents):
                    drone_id = f'drone_{i}'
                    state = obs[drone_id]
                    
                    # 使用MAPPO智能体选择动作
                    if hasattr(agent, 'select_action'):
                        # 离散动作：select_action 返回动作索引 (0-26)
                        action = agent.select_action(state, training=False)
                        # 动作索引直接传给环境，环境会映射到速度
                    else:
                        # 如果没有训练好的模型，使用随机动作索引
                        action = np.random.randint(0, self.drone_env.num_actions)
                    
                    actions[drone_id] = action
                
                # 执行动作
                next_obs, rewards, terminated, truncated, infos = self.drone_env.step(actions)
                
                # 更新状态
                obs = next_obs
                total_reward += sum(rewards.values())
                step_count += 1
                
                # 更新UI显示
                self._update_drone_ui_data()
                
                if all(terminated.values()) or all(truncated.values()):
                    break
                
                time.sleep(0.5)  # 短暂延迟以允许UI更新
            
            backend_logger.info(f"无人机任务完成! 总奖励: {total_reward:.2f}")
            self.is_inferring = False
            
        except Exception as e:
            backend_logger.error(f"无人机任务执行出错: {str(e)}")
            self.system_error.emit(f"无人机任务执行出错: {str(e)}")
            self.is_inferring = False
    
    def _update_drone_ui_data(self):
        """更新无人机UI数据"""
        drone_data = []
        
        if self.drone_env is not None:
            for i in range(self.drone_env.num_drones):
                drone_data.append({
                    'id': i,
                    'position': self.drone_env.drone_positions[i].tolist(),
                    'battery': self.drone_env.drone_batteries[i],
                    'speed': np.linalg.norm(self.drone_env.drone_velocities[i]),
                    'task_progress': self.drone_env.waypoints_visited / self.drone_env.num_waypoints if self.drone_env.task_type == 'inspection' else 1.0 - self.drone_env.formation_error,
                    'status': 'flying' if np.linalg.norm(self.drone_env.drone_velocities[i]) > 0.1 else 'idle'
                })
            
            # 添加任务信息
            task_info = {
                'task_type': self.drone_env.task_type,
                'drones': drone_data
            }
            
            # 巡检任务信息
            if self.drone_env.task_type == 'inspection':
                task_info['start_point'] = self.drone_env.start_point.tolist() if self.drone_env.start_point is not None else None
                task_info['end_point'] = self.drone_env.end_point.tolist() if self.drone_env.end_point is not None else None
                task_info['waypoints'] = [wp.tolist() for wp in self.drone_env.waypoints]
                task_info['inspection_path'] = [wp.tolist() for wp in self.drone_env.inspection_path]
                task_info['current_path_index'] = self.drone_env.current_path_index
                task_info['is_return_trip'] = self.drone_env.is_return_trip
                task_info['waypoints_visited'] = self.drone_env.waypoints_visited
            
            # 队形任务信息
            elif self.drone_env.task_type == 'formation':
                task_info['formation_type'] = self.drone_env.formation_type
                task_info['formation_start'] = self.drone_env.formation_start.tolist() if hasattr(self.drone_env, 'formation_start') else None
                task_info['formation_end'] = self.drone_env.formation_end.tolist() if hasattr(self.drone_env, 'formation_end') else None
                task_info['formation_error'] = self.drone_env.formation_error
                task_info['leader_drone_idx'] = self.drone_env.leader_drone_idx
                task_info['formation_offsets'] = [offset.tolist() for offset in self.drone_env.formation_offsets]
            
            # 协同包围任务信息
            elif self.drone_env.task_type == 'encirclement':
                task_info['target_position'] = self.drone_env.target_position.tolist() if hasattr(self.drone_env, 'target_position') else [50, 50, 25]
                task_info['target_velocity'] = self.drone_env.target_velocity.tolist() if hasattr(self.drone_env, 'target_velocity') else [0.3, 0.3, 0.0]
                task_info['encirclement_radius'] = self.drone_env.encirclement_radius
                task_info['encirclement_time'] = self.drone_env.encirclement_time
                task_info['encirclement_success'] = self.drone_env.encirclement_success
        
        # 更新UI
        try:
            self.drone_data_updated.emit(task_info if self.drone_env is not None else {})
        except RuntimeError:
            self.is_inferring = False
    
    def reset_drone_mission(self):
        """重置无人机任务"""
        self.is_inferring = False
        
        if self.drone_env is not None:
            obs, _ = self.drone_env.reset()
            backend_logger.info("无人机任务已重置")
            
            # 发送重置信号
            try:
                self.drone_data_updated.emit([])
            except RuntimeError:
                pass
    
    def update_drone_positions(self, drones_data):
        """更新无人机的起点和终点位置"""
        if self.drone_env is not None:
            for i, drone_data in enumerate(drones_data):
                if i < self.drone_env.num_drones:
                    # 更新起点位置
                    start_pos = drone_data.get('start_position', drone_data.get('position', [0, 0, 0]))
                    self.drone_env.drone_positions[i] = np.array(start_pos, dtype=np.float64)
                    
                    # 存储起点和终点位置
                    if not hasattr(self.drone_env, 'start_positions'):
                        self.drone_env.start_positions = np.zeros((self.drone_env.num_drones, 3))
                    if not hasattr(self.drone_env, 'end_positions'):
                        self.drone_env.end_positions = np.zeros((self.drone_env.num_drones, 3))
                    
                    self.drone_env.start_positions[i] = np.array(start_pos, dtype=np.float64)
                    
                    # 更新终点位置
                    end_pos = drone_data.get('end_position', [100, 100, 30])
                    self.drone_env.end_positions[i] = np.array(end_pos, dtype=np.float64)
            
            backend_logger.info(f"无人机位置已更新: {len(drones_data)} 台无人机")
            
            # 发送更新信号
            try:
                self.drone_data_updated.emit(drones_data)
            except RuntimeError:
                pass
        else:
            backend_logger.warning("无人机环境未初始化，无法更新位置")