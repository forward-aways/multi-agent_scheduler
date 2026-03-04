"""
多智能体无人机调度环境
用于训练多智能体协作的无人机巡检和队形保持系统
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, List, Union
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import ProjectLogger

env_logger = ProjectLogger('drone_env', log_dir='logs')


class MultiAgentDroneEnv(gym.Env):
    """
    多智能体无人机调度环境
    每个无人机作为一个智能体
    支持巡检任务和队形保持任务
    """
    
    def __init__(self, config: Dict[str, Any], custom_positions: Dict[str, Any] = None):
        """
        初始化环境
        
        参数:
            config: 环境配置参数
            custom_positions: 用户自定义位置（推理时使用）
                {
                    'task_type': 'inspection' or 'formation',
                    'start_point': [x, y, z],
                    'end_point': [x, y, z],
                    'waypoints': [[x, y, z], ...]  # 仅巡检任务
                }
        """
        super().__init__()
        
        # 无人机配置
        self.num_drones = config.get('num_drones', 3)
        self.max_speed = config.get('max_speed', 10.0)  # m/s
        self.battery_capacity = config.get('battery_capacity', 100.0)
        self.payload_capacity = config.get('payload_capacity', 5.0)
        
        # 空间配置
        self.space_size = config.get('space_size', [100, 100, 50])  # x, y, z
        
        # 任务配置
        self.task_type = config.get('task_type', 'inspection')  # 'inspection', 'formation', or 'search'
        self.max_steps = config.get('max_steps', 200)
        
        # 巡检任务配置
        self.num_waypoints = config.get('num_waypoints', 4)
        self.waypoints = []
        self.start_point = None  # 起点
        self.end_point = None  # 终点
        self.inspection_path = []  # 完整的巡检路径（包含往返）
        self.current_path_index = 0  # 当前路径点索引
        self.waypoints_visited = 0
        self.is_return_trip = False  # 是否在返航阶段
        
        # 协同搜索任务配置
        self.num_targets = config.get('num_targets', 5)  # 目标数量
        self.search_area_size = config.get('search_area_size', [100, 100, 50])  # 搜索区域大小
        self.targets = []  # 目标位置列表
        self.discovered_targets = []  # 已发现的目标
        self.target_discovery_radius = 10.0  # 目标发现半径
        self.search_regions = []  # 每个无人机的搜索区域
        self.search_coverage = np.zeros((self.num_drones,))  # 每个无人机的搜索覆盖率
        self.team_search_progress = 0.0  # 团队搜索进度
        
        # 队形任务配置
        self.formation_type = config.get('formation_type', 'triangle')
        # 队形定义：相对于领航机的偏移量（无人机0作为领航机）
        self.formations = {
            'triangle': [(0, 0, 0), (-10, -10, 0), (10, -10, 0)],  # 三角形
            'v_shape': [(0, 0, 0), (-10, 10, 0), (-10, -10, 0)],   # V形
            'line': [(0, 0, 0), (10, 0, 0), (20, 0, 0)]           # 一字形
        }
        self.formation_offsets = []  # 队形偏移量
        self.leader_drone_idx = 0  # 领航机索引（无人机0）
        self.formation_start = None  # 队形起点
        self.formation_end = None  # 队形终点
        self.formation_completed = False  # 队形任务完成标志
        self.formation_error = 0.0  # 队形误差
        
        # 无人机状态
        self.drone_positions = np.zeros((self.num_drones, 3))
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.drone_batteries = np.ones(self.num_drones) * self.battery_capacity
        self.drone_payloads = np.zeros(self.num_drones)
        
        # 任务状态
        self.current_waypoint = 0
        self.waypoints_visited = 0
        self.formation_error = 0.0
        self.task_completed = False
        
        # 时间跟踪
        self.current_step = 0
        
        # 保存自定义位置（推理时使用）
        self.custom_positions = custom_positions
        
        # 为每个无人机定义动作空间和观测空间
        self.action_spaces = {}
        self.observation_spaces = {}
        
        # 定义27个离散动作（3x3x3的速度组合）
        # 每个维度：-1, 0, 1
        self.discrete_actions = []
        for vx in [-1, 0, 1]:
            for vy in [-1, 0, 1]:
                for vz in [-1, 0, 1]:
                    self.discrete_actions.append([
                        vx * self.max_speed,
                        vy * self.max_speed,
                        vz * self.max_speed
                    ])
        self.num_actions = len(self.discrete_actions)  # 27
        
        # 调试信息
        env_logger.info(f"离散动作空间: max_speed={self.max_speed}, num_actions={self.num_actions}")
        env_logger.info(f"动作示例: {self.discrete_actions[0]}, {self.discrete_actions[13]}, {self.discrete_actions[26]}")
        
        for i in range(self.num_drones):
            # 每个无人机的动作空间：离散动作（27个选择）
            self.action_spaces[f'drone_{i}'] = gym.spaces.Discrete(self.num_actions)
            
            # 根据任务类型计算观测空间维度（参考服务器环境设计）
            # 本地状态：位置3 + 速度3 + 电量1 = 7维
            # 其他无人机相对位置：3 * (n-1) = 6维（3个无人机时）
            local_dim = 3 + 3 + 1  # 位置+速度+电量
            neighbor_dim = 3 * (self.num_drones - 1)  # 相对位置
            
            if self.task_type == 'inspection':
                # 巡检任务：本地7 + 相对6 + 目标点3 + 进度1 = 17维
                task_dim = 3 + 1  # 当前目标点3 + 路径进度1
                obs_dim = local_dim + neighbor_dim + task_dim
            elif self.task_type == 'formation':
                # 队形任务：本地7 + 相对6 + 领航机3 + 终点3 + 偏移3 + 误差1 = 23维
                task_dim = 3 + 3 + 3 + 1  # 领航机3 + 终点3 + 偏移3 + 误差1
                obs_dim = local_dim + neighbor_dim + task_dim
            elif self.task_type == 'encirclement':
                # 包围任务：本地7 + 相对6 + 目标3 + 目标速度3 + 其他无人机包围状态3 + 包围时间1 = 23维
                task_dim = 3 + 3 + 3 + 1  # 目标位置3 + 目标速度3 + 其他无人机包围状态3 + 包围时间1
                obs_dim = local_dim + neighbor_dim + task_dim
            else:
                obs_dim = local_dim + neighbor_dim
            
            self.observation_spaces[f'drone_{i}'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        返回:
            初始观测和信息
        """
        super().reset(seed=seed)
        
        # 重置时间
        self.current_step = 0
        self.task_completed = False
        
        # 先初始化任务（获取起点、终点、目标位置等）
        if self.task_type == 'inspection':
            self._init_inspection_task()
        elif self.task_type == 'formation':
            self._init_formation_task()
        elif self.task_type == 'encirclement':
            self._init_encirclement_task()
        
        # 根据任务类型初始化无人机位置
        if self.task_type == 'formation':
            # 编队任务：按队形要求初始化位置
            self._init_formation_positions()
        elif self.task_type == 'encirclement':
            # 围捕任务：在目标周围初始化
            self._init_encirclement_positions()
        else:
            # 巡检任务：随机分布
            self.drone_positions = np.random.rand(self.num_drones, 3) * np.array(self.space_size)
        
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.drone_batteries = np.ones(self.num_drones) * self.battery_capacity
        self.drone_payloads = np.zeros(self.num_drones)
        
        env_logger.info(f"环境重置完成，任务类型: {self.task_type}")
        
        
        return self._get_observations(), {}
    
    def _init_inspection_task(self):
        """初始化巡检任务（完全重构版）"""
        # 每个无人机独立的路径点索引
        self.drone_path_indices = np.zeros(self.num_drones, dtype=int)
        self.waypoints_visited = 0
        
        # 记录前一帧的路径点索引，用于计算奖励
        self._prev_drone_path_indices = np.zeros(self.num_drones, dtype=int)
        
        # 检查是否有用户自定义位置
        if self.custom_positions is not None and self.custom_positions.get('task_type') == 'inspection':
            # 使用用户自定义位置（推理时）
            self.start_point = np.array(self.custom_positions['start_point'])
            self.end_point = np.array(self.custom_positions['end_point'])
            if 'waypoints' in self.custom_positions:
                self.waypoints = [np.array(wp) for wp in self.custom_positions['waypoints']]
            env_logger.info("使用用户自定义位置（推理模式）")
        else:
            # 使用固定位置（训练时）
            # 起点：左下角
            self.start_point = np.array([15.0, 15.0, 15.0])
            
            # 终点：右上角
            self.end_point = np.array([85.0, 85.0, 35.0])
            
            # 检查点：固定在路径上
            self.waypoints = [
                np.array([30.0, 30.0, 20.0]),  # 检查点1
                np.array([50.0, 50.0, 25.0]),  # 检查点2
                np.array([70.0, 70.0, 30.0]),  # 检查点3
                np.array([40.0, 60.0, 22.0])   # 检查点4
            ]
            
            env_logger.info("使用固定位置（训练模式）")
        
        # 构建简化路径：起点 → A → B → C → D → 终点（单向）
        self.inspection_path = []
        self.inspection_path.append(self.start_point)  # 起点
        
        # 去程：起点 → A → B → C → D → 终点
        for waypoint in self.waypoints:
            self.inspection_path.append(waypoint)
        self.inspection_path.append(self.end_point)  # 终点
        
        self.is_return_trip = False  # 不需要返航
        
        env_logger.info(f"巡检任务初始化（完全重构版），检查点数量: {self.num_waypoints}")
        env_logger.info(f"起点: {self.start_point}, 终点: {self.end_point}")
        env_logger.info(f"检查点: {self.waypoints}")
        env_logger.info(f"总路径点数: {len(self.inspection_path)} (单向)")
    
    def _init_encirclement_task(self):
        """初始化协同包围任务（多智能体协作包围静态目标）"""
        # 目标位置（静态目标）
        self.target_position = np.array([50.0, 50.0, 25.0])  # 目标位置
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # 静态目标，速度为0
        self.encirclement_radius = 20.0  # 包围半径
        self.encirclement_completed = False  # 包围是否完成
        self.encirclement_time = 0  # 包围持续时间
        
        # 检查是否有用户自定义位置
        if self.custom_positions is not None and self.custom_positions.get('task_type') == 'encirclement':
            # 使用用户自定义位置（推理时）
            if 'target_position' in self.custom_positions:
                self.target_position = np.array(self.custom_positions['target_position'])
            env_logger.info("使用用户自定义位置（推理模式）")
        else:
            # 使用固定位置（训练时）
            env_logger.info("使用固定位置（训练模式）")
        
        # 初始化包围状态
        self.encirclement_success = False  # 是否成功包围
        self.encirclement_steps = 0  # 包围步数
        
        env_logger.info(f"协同包围任务初始化（静态目标），包围半径: {self.encirclement_radius}")
        env_logger.info(f"目标位置: {self.target_position}")
    
    def _init_formation_task(self):
        """初始化队形任务（给定起点和终点，无人机自主规划路径）"""
        # 检查是否有用户自定义位置
        if self.custom_positions is not None and self.custom_positions.get('task_type') == 'formation':
            # 使用用户自定义位置（推理时）
            self.formation_start = np.array(self.custom_positions['start_point'])
            self.formation_end = np.array(self.custom_positions['end_point'])
            env_logger.info("使用用户自定义位置（推理模式）")
        else:
            # 随机生成起点和终点（训练时）
            self.formation_start = np.array([
                np.random.uniform(10, 30),
                np.random.uniform(10, 30),
                np.random.uniform(10, 20)
            ])
            self.formation_end = np.array([
                np.random.uniform(70, 90),
                np.random.uniform(70, 90),
                np.random.uniform(30, 45)
            ])
            env_logger.info("随机生成位置（训练模式）")
        
        # 获取队形偏移量
        formation_offsets = self.formations.get(self.formation_type, self.formations['triangle'])
        self.formation_offsets = [np.array(offset) for offset in formation_offsets]
        
        # 如果无人机数量超过队形定义的数量，动态扩展偏移量
        if self.num_drones > len(self.formation_offsets):
            env_logger.warning(f"无人机数量({self.num_drones})超过队形定义({len(self.formation_offsets)})，动态扩展偏移量")
            base_offsets = self.formation_offsets.copy()
            while len(self.formation_offsets) < self.num_drones:
                # 为额外的无人机生成偏移量（在现有队形基础上扩展）
                extra_idx = len(self.formation_offsets)
                # 使用最后一个偏移量作为基础，添加额外间距
                last_offset = base_offsets[-1] if len(base_offsets) > 0 else np.array([0, 0, 0])
                extra_offset = last_offset + np.array([10 * (extra_idx - len(base_offsets) + 1), 0, 0])
                self.formation_offsets.append(extra_offset)
            env_logger.info(f"扩展后的队形偏移量: {self.formation_offsets}")
        
        # 初始化队形误差
        self.formation_error = 0.0
        
        # 任务完成标志（领航机到达终点）
        self.formation_completed = False
        
        # 初始化领航机距离终点的上一步距离（用于计算移动奖励）
        leader_pos = self.drone_positions[self.leader_drone_idx]
        self._prev_leader_distance_to_end = np.linalg.norm(leader_pos - self.formation_end)
        
        # 初始化领航机上一步位置（用于计算跟随奖励）
        self._prev_leader_pos = leader_pos.copy()
        
        # 初始化每个无人机的上一步位置（用于计算跟随奖励）
        self._prev_positions = [pos.copy() for pos in self.drone_positions]
        
        env_logger.info(f"队形任务初始化，队形类型: {self.formation_type}")
        env_logger.info(f"领航机: 无人机{self.leader_drone_idx}")
        env_logger.info(f"队形偏移量: {self.formation_offsets}")
        env_logger.info(f"起点: {self.formation_start}, 终点: {self.formation_end}")
        env_logger.info("无人机将自主规划路径，只要保持队形即可")
    
    def _init_formation_positions(self):
        """初始化编队任务无人机位置（按队形要求）"""
        # 领航机在 formation_start
        self.drone_positions = np.zeros((self.num_drones, 3))
        self.drone_positions[self.leader_drone_idx] = self.formation_start.copy()
        
        # 其他无人机按偏移量放置
        for i in range(self.num_drones):
            if i != self.leader_drone_idx:
                self.drone_positions[i] = self.formation_start + self.formation_offsets[i]
        
        # 确保位置在空间范围内
        self.drone_positions = np.clip(
            self.drone_positions,
            [0, 0, 0],
            self.space_size
        )
        
        env_logger.info(f"编队位置初始化完成: {self.drone_positions}")
    
    def _init_encirclement_positions(self):
        """初始化围捕任务无人机位置（在目标周围）"""
        self.drone_positions = np.zeros((self.num_drones, 3))
        
        # 在目标周围均匀分布
        angles = np.linspace(0, 2 * np.pi, self.num_drones, endpoint=False)
        radius = self.encirclement_radius * 1.5  # 在包围圈外一点
        
        for i, angle in enumerate(angles):
            self.drone_positions[i] = self.target_position + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                np.random.uniform(-5, 5)  # Z轴有一点随机性
            ])
        
        # 确保位置在空间范围内
        self.drone_positions = np.clip(
            self.drone_positions,
            [0, 0, 0],
            self.space_size
        )
        
        env_logger.info(f"围捕位置初始化完成: {self.drone_positions}")
    
    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        执行一步
        
        参数:
            actions: 每个无人机的动作 {drone_id: action_index}，action_index是0-26的整数
            
        返回:
            observations, rewards, terminated, truncated, infos
        """
        # 执行动作
        for i in range(self.num_drones):
            drone_id = f'drone_{i}'
            if drone_id in actions:
                action_idx = actions[drone_id]
                
                # 将离散动作索引转换为连续速度
                if isinstance(action_idx, (int, np.integer)):
                    velocity = np.array(self.discrete_actions[action_idx], dtype=np.float32)
                else:
                    # 兼容旧版本，如果是数组则直接使用
                    velocity = np.array(action_idx, dtype=np.float32)
                    velocity = np.clip(velocity, -self.max_speed, self.max_speed)
                
                # 更新速度
                self.drone_velocities[i] = velocity
                
                # 更新位置
                self.drone_positions[i] += velocity
                
                # 限制在空间范围内
                self.drone_positions[i] = np.clip(
                    self.drone_positions[i], 
                    [0, 0, 0], 
                    self.space_size
                )
                
                # 消耗电量
                speed = np.linalg.norm(velocity)
                energy_cost = speed * 0.01  # 能耗系数
                self.drone_batteries[i] = max(0, self.drone_batteries[i] - energy_cost)
        
        # 更新任务状态
        if self.task_type == 'inspection':
            self._update_inspection_task()
        elif self.task_type == 'formation':
            self._update_formation_task()
        elif self.task_type == 'encirclement':
            self._update_encirclement_task()
        
        # 计算奖励
        rewards = self._compute_rewards()
        
        # 检查是否结束
        terminated = {f'drone_{i}': False for i in range(self.num_drones)}
        truncated = {f'drone_{i}': False for i in range(self.num_drones)}
        
        self.current_step += 1
        
        # 检查任务完成
        if self.task_completed:
            for i in range(self.num_drones):
                terminated[f'drone_{i}'] = True
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            for i in range(self.num_drones):
                truncated[f'drone_{i}'] = True
        
        # 检查电量耗尽
        for i in range(self.num_drones):
            if self.drone_batteries[i] <= 0:
                terminated[f'drone_{i}'] = True
        
        # 获取观测
        observations = self._get_observations()
        
        # 信息
        infos = {}
        for i in range(self.num_drones):
            task_progress = 0.0
            if self.task_type == 'inspection':
                task_progress = self.waypoints_visited / self.num_waypoints
            elif self.task_type == 'formation':
                task_progress = 1.0 - self.formation_error
            elif self.task_type == 'encirclement':
                task_progress = self.encirclement_time / 50.0  # 包围进度（50步为完成）
            
            infos[f'drone_{i}'] = {
                'battery': self.drone_batteries[i],
                'position': self.drone_positions[i],
                'task_progress': task_progress
            }
        
        return observations, rewards, terminated, truncated, infos
    
    def _update_inspection_task(self):
        """更新巡检任务状态（协作版：多智能体协同访问检查点）"""
        # 检查所有检查点是否都被访问过
        all_waypoints_visited = True
        
        for i in range(self.num_drones):
            path_idx = self.drone_path_indices[i]
            
            # 检查是否完成所有路径点
            if path_idx >= len(self.inspection_path):
                continue  # 该无人机已完成所有路径点
            
            all_waypoints_visited = False
            
            target = self.inspection_path[path_idx]
            distance = np.linalg.norm(self.drone_positions[i] - target)
            
            # 检查是否到达当前路径点
            # 使用更大的到达半径以避免因微小波动而反复触发
            if distance < 5.0 and not hasattr(self, f'_reached_waypoint_{i}_{path_idx}'):
                # 标记该无人机已到达此路径点，避免重复触发
                setattr(self, f'_reached_waypoint_{i}_{path_idx}', True)
                self.drone_path_indices[i] = min(path_idx + 1, len(self.inspection_path))
                self.waypoints_visited += 1
                env_logger.debug(f"无人机{i}到达路径点{path_idx}，前进到路径点{self.drone_path_indices[i]}/{len(self.inspection_path)}")
        
        # 检查是否所有路径点都被至少一个无人机访问过
        if all_waypoints_visited:
            self.task_completed = True
            env_logger.info("巡检任务完成（所有检查点已被访问）")
    
    def _update_encirclement_task(self):
        """更新协同包围任务状态（多智能体协作包围移动目标）"""
        # 更新目标位置（移动目标）
        self.target_position += self.target_velocity
        
        # 边界检查：如果目标到达边界，反弹
        for i in range(3):
            if self.target_position[i] < 0 or self.target_position[i] > self.space_size[i]:
                self.target_velocity[i] *= -1  # 反转速度
        
        # 检查是否所有无人机都在包围半径内
        all_in_radius = True
        for i in range(self.num_drones):
            distance = np.linalg.norm(self.drone_positions[i] - self.target_position)
            if distance > self.encirclement_radius:
                all_in_radius = False
                break
        
        # 如果所有无人机都在包围半径内，增加包围时间
        if all_in_radius:
            self.encirclement_time += 1
            self.encirclement_steps += 1
            
            # 如果包围时间超过阈值，认为包围成功
            if self.encirclement_time >= 50:  # 需要持续50步
                self.encirclement_success = True
                self.task_completed = True
                env_logger.info("协同包围任务完成（目标已被成功包围）")
        else:
            self.encirclement_time = 0  # 重置包围时间
    
    def _update_formation_task(self):
        """更新队形任务状态（给定起点和终点，无人机自主规划路径）"""
        # 检查领航机是否到达终点
        leader_pos = self.drone_positions[self.leader_drone_idx]
        distance_to_end = np.linalg.norm(leader_pos - self.formation_end)
        
        # 如果领航机到达终点，任务完成
        if distance_to_end < 5.0:  # 到达阈值
            self.formation_completed = True
            self.task_completed = True
            env_logger.info(f"队形任务完成，领航机已到达终点")
        
        # 计算队形误差（基于相对位置）
        total_error = 0.0
        
        for i in range(self.num_drones):
            # 所有无人机都计算相对于期望位置的误差
            expected_pos = leader_pos + self.formation_offsets[i]
            actual_pos = self.drone_positions[i]
            distance = np.linalg.norm(actual_pos - expected_pos)
            # 使用固定阈值归一化（20米作为最大可接受误差）
            normalized_distance = min(1.0, distance / 20.0)
            total_error += normalized_distance
        
        self.formation_error = total_error / self.num_drones
        
        # 更新上一步位置（用于计算移动奖励）
        self._prev_leader_pos = leader_pos.copy()
        self._prev_positions = [pos.copy() for pos in self.drone_positions]
    
    def _compute_rewards(self) -> Dict[str, float]:
        """
        计算奖励
        
        返回:
            每个无人机的奖励 {drone_id: reward}
        """
        rewards = {}
        
        for i in range(self.num_drones):
            drone_id = f'drone_{i}'
            reward = 0.0
            
            if self.task_type == 'inspection':
                reward = self._compute_inspection_reward(i)
            elif self.task_type == 'formation':
                reward = self._compute_formation_reward(i)
            elif self.task_type == 'encirclement':
                reward = self._compute_encirclement_reward(i)
            
            # 通用奖励
            reward += self._compute_general_reward(i)
            
            rewards[drone_id] = reward
        
        return rewards
    
    def _compute_inspection_reward(self, drone_idx: int) -> float:
        """
        计算巡检任务奖励（简化版 - 参考服务器调度成功的设计）
        
        核心设计原则：
        1. 简单明确：奖励信号清晰，不复杂
        2. 即时反馈：每一步都有明确的正负反馈
        3. 任务导向：围绕完成任务设计奖励
        4. 去耦合：每个智能体只对自己的行为负责
        """
        reward = 0.0
        current_pos = self.drone_positions[drone_idx]
        path_idx = self.drone_path_indices[drone_idx]
        
        # 1. 完成任务奖励（核心奖励）
        if path_idx >= len(self.inspection_path):
            reward = 100.0  # 完成所有路径点给予高奖励
            return reward
        
        # 2. 接近目标奖励
        target = self.inspection_path[path_idx]
        distance_to_target = np.linalg.norm(current_pos - target)
        prev_distance = getattr(self, f'_drone_{drone_idx}_prev_distance', distance_to_target + 1.0)
        distance_moved = prev_distance - distance_to_target
        
        if distance_moved > 0:
            reward += distance_moved * 2.0  # 向目标移动给予奖励
        
        # 更新距离
        setattr(self, f'_drone_{drone_idx}_prev_distance', distance_to_target)
        
        # 3. 到达路径点奖励
        if distance_to_target < 5.0:
            reward += 10.0  # 到达路径点给予奖励
        
        # 4. 速度奖励（鼓励移动）
        velocity = self.drone_velocities[drone_idx]
        speed = np.linalg.norm(velocity)
        
        if speed > 0.5:
            reward += 1.0  # 保持移动给予奖励
        else:
            reward -= 0.5  # 静止给予轻微惩罚
        
        # 5. 避免碰撞奖励
        for i in range(self.num_drones):
            if i != drone_idx:
                distance = np.linalg.norm(current_pos - self.drone_positions[i])
                if distance < 3.0:
                    reward -= 3.0  # 距离太近惩罚
        
        return reward
    
    def _is_waypoint_visited_by_others(self, current_drone_idx, waypoint_idx):
        """检查指定路径点是否被其他无人机访问过"""
        if waypoint_idx < 0:
            return False
        for other_idx in range(self.num_drones):
            if other_idx != current_drone_idx:
                if self.drone_path_indices[other_idx] > waypoint_idx:
                    return True
        return False

    def _dist_to_segment(self, point, seg_start, seg_end):
        """计算点到线段的最短距离"""
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq == 0:
            return np.linalg.norm(point - seg_start)
        
        t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
        projection = seg_start + t * seg_vec
        return np.linalg.norm(point - projection)
    
    def _compute_encirclement_reward(self, drone_idx: int) -> float:
        """
        计算协同包围任务奖励（调整版 - 增加基础奖励、连续奖励和边界惩罚）
        """
        reward = 0.1  # 基础存活奖励
        current_pos = self.drone_positions[drone_idx]
        
        # 接近目标奖励（根据移动距离给予奖励）
        distance_to_target = np.linalg.norm(current_pos - self.target_position)
        prev_distance = getattr(self, f'_drone_{drone_idx}_prev_distance', distance_to_target + 1.0)
        distance_moved = prev_distance - distance_to_target
        
        # 根据移动距离给予奖励，最大2.0
        reward += np.clip(distance_moved * 2.0, -1.0, 2.0)
        
        setattr(self, f'_drone_{drone_idx}_prev_distance', distance_to_target)
        
        # 距离目标越近奖励越大，最大3.0
        reward += max(0, (100 - distance_to_target) / 33)
        
        # 在包围半径内奖励
        if distance_to_target <= self.encirclement_radius:
            reward += 2.0  # 在包围半径内给予奖励
        
        # 包围进度奖励
        if self.encirclement_time > 0:
            reward += 2.0  # 包围时间奖励
        
        # 任务完成奖励
        if self.encirclement_success:
            reward += 100.0
        
        # 边界惩罚 - 防止无人机卡在边界
        margin = 5.0  # 边界安全距离
        for i in range(3):  # x, y, z
            if current_pos[i] < margin:
                reward -= (margin - current_pos[i]) * 0.5  # 靠近下边界惩罚
            elif current_pos[i] > self.space_size[i] - margin:
                reward -= (current_pos[i] - (self.space_size[i] - margin)) * 0.5  # 靠近上边界惩罚
        
        return reward
    
    def _compute_formation_reward(self, drone_idx: int) -> float:
        """
        计算队形任务奖励（精度优化版 - 重点优化队形保持精度）
        """
        reward = 1.0  # 增加基础存活奖励
        leader_pos = self.drone_positions[self.leader_drone_idx]
        current_pos = self.drone_positions[drone_idx]
        
        if drone_idx == self.leader_drone_idx:
            # ==================== 领航机奖励 ====================
            # 向终点移动奖励（根据移动距离给予奖励）
            distance_to_end = np.linalg.norm(current_pos - self.formation_end)
            prev_distance = getattr(self, '_prev_leader_distance', distance_to_end + 1.0)
            distance_moved = prev_distance - distance_to_end
            
            # 根据移动距离给予奖励，最大2.0
            reward += np.clip(distance_moved * 2.0, -1.0, 2.0)
            
            self._prev_leader_distance = distance_to_end
            
            # 距离终点越近奖励越大
            reward += max(0, (100 - distance_to_end) / 50)  # 最大2.0
            
            # 到达终点奖励
            if distance_to_end < 5.0:
                reward += 100.0
            
        else:
            # ==================== 跟随者奖励（精度优先）====================
            # 计算期望位置
            expected_pos = leader_pos + self.formation_offsets[drone_idx]
            distance_to_expected = np.linalg.norm(current_pos - expected_pos)
            
            # ===== 核心：高精度队形奖励（误差<1米）=====
            # 使用指数衰减奖励，距离越近奖励越高
            if distance_to_expected < 0.5:
                reward += 50.0  # 极精确（<0.5米），最高奖励
            elif distance_to_expected < 1.0:
                reward += 40.0 + (1.0 - distance_to_expected) * 20.0  # 40-50
            elif distance_to_expected < 2.0:
                reward += 25.0 + (2.0 - distance_to_expected) * 15.0  # 25-40
            elif distance_to_expected < 5.0:
                reward += 10.0 + (5.0 - distance_to_expected) * 5.0  # 10-25
            else:
                reward += max(-20, 10.0 - distance_to_expected * 2.0)  # 负奖励，惩罚大误差
            
            # ===== 高精度相对位置奖励（误差<1米）=====
            relative_pos = current_pos - leader_pos
            expected_relative = self.formation_offsets[drone_idx]
            relative_error = np.linalg.norm(relative_pos - expected_relative)
            
            # 相对位置误差奖励（高精度要求）
            if relative_error < 0.5:
                reward += 40.0  # 极精确
            elif relative_error < 1.0:
                reward += 30.0 + (1.0 - relative_error) * 20.0  # 30-40
            elif relative_error < 2.0:
                reward += 15.0 + (2.0 - relative_error) * 15.0  # 15-30
            elif relative_error < 5.0:
                reward += 5.0 + (5.0 - relative_error) * 3.33  # 5-15
            else:
                reward += max(-15, 5.0 - relative_error * 1.5)  # 负奖励
            
            # ===== 严格队形维持惩罚（误差>3米开始惩罚）=====
            if distance_to_expected > 3.0:
                reward -= (distance_to_expected - 3.0) ** 2 * 0.5  # 平方惩罚，误差越大惩罚越重
            
            # ===== 速度匹配奖励（与领航机速度同步）=====
            leader_vel = self.drone_velocities[self.leader_drone_idx]
            my_vel = self.drone_velocities[drone_idx]
            velocity_diff = np.linalg.norm(my_vel - leader_vel)
            if velocity_diff < 0.5:
                reward += 10.0  # 速度同步良好
            elif velocity_diff < 1.0:
                reward += 5.0
            else:
                reward -= velocity_diff * 0.5  # 速度不同步惩罚
            
            # ===== 跟随移动奖励（适度）=====
            prev_pos = getattr(self, f'_drone_{drone_idx}_prev_pos', current_pos)
            leader_prev_pos = getattr(self, '_prev_leader_pos', leader_pos)
            
            leader_movement = np.linalg.norm(leader_pos - leader_prev_pos)
            my_movement = np.linalg.norm(current_pos - prev_pos)
            
            if leader_movement > 0.1 and my_movement > 0.1:
                reward += 1.0  # 适度跟随奖励
            
            setattr(self, f'_drone_{drone_idx}_prev_pos', current_pos)
        
        # 保存领航机位置用于下一帧计算
        setattr(self, '_prev_leader_pos', leader_pos)
        
        # 任务完成奖励
        if self.task_completed:
            reward += 150.0
        
        # 边界惩罚（避免无人机飞出边界）
        for dim in range(3):
            if current_pos[dim] < 3 or current_pos[dim] > self.space_size[dim] - 3:
                reward -= 1.0
        
        return reward
    
    def _compute_general_reward(self, drone_idx: int) -> float:
        """计算通用奖励（简化版，返回0，避免干扰）"""
        return 0.0
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        获取观测（参考服务器环境设计）
        
        返回:
            每个无人机的观测 {drone_id: observation}
        """
        observations = {}
        
        for i in range(self.num_drones):
            drone_id = f'drone_{i}'
            
            # 本地状态（归一化）
            local_state = np.array([
                self.drone_positions[i][0] / self.space_size[0],  # x位置
                self.drone_positions[i][1] / self.space_size[1],  # y位置
                self.drone_positions[i][2] / self.space_size[2],  # z位置
                (self.drone_velocities[i][0] + self.max_speed) / (2 * self.max_speed),  # x速度（归一化到[0,1]）
                (self.drone_velocities[i][1] + self.max_speed) / (2 * self.max_speed),  # y速度
                (self.drone_velocities[i][2] + self.max_speed) / (2 * self.max_speed),  # z速度
                self.drone_batteries[i] / self.battery_capacity  # 电量比例
            ])
            
            # 其他无人机相对位置
            neighbor_states = []
            for j in range(self.num_drones):
                if i != j:
                    relative_pos = (self.drone_positions[j] - self.drone_positions[i]) / np.array(self.space_size)
                    neighbor_states.extend(relative_pos)
            
            # 任务信息
            if self.task_type == 'inspection':
                path_idx = self.drone_path_indices[i]
                
                # 当前目标点（归一化）
                if path_idx < len(self.inspection_path):
                    target = self.inspection_path[path_idx]
                    target_state = np.array([
                        target[0] / self.space_size[0],
                        target[1] / self.space_size[1],
                        target[2] / self.space_size[2]
                    ])
                else:
                    target_state = np.array([0.0, 0.0, 0.0])
                
                # 路径进度（归一化）
                progress = path_idx / max(1, len(self.inspection_path))
                
                # 组合观测
                obs = np.concatenate([local_state, neighbor_states, target_state, [progress]])
                observations[drone_id] = obs.astype(np.float32)
            
            elif self.task_type == 'formation':
                # 队形任务观测
                leader_pos = self.drone_positions[self.leader_drone_idx]
                
                # 领航机位置（归一化）
                leader_state = np.array([
                    leader_pos[0] / self.space_size[0],
                    leader_pos[1] / self.space_size[1],
                    leader_pos[2] / self.space_size[2]
                ])
                
                # 终点位置（归一化）
                end_state = np.array([
                    self.formation_end[0] / self.space_size[0],
                    self.formation_end[1] / self.space_size[1],
                    self.formation_end[2] / self.space_size[2]
                ])
                
                # 相对偏移（归一化）
                offset_state = self.formation_offsets[i] / np.array(self.space_size)
                
                # 队形误差（归一化）
                error_state = np.array([1.0 - self.formation_error / 20.0])
                
                # 组合观测
                obs = np.concatenate([local_state, neighbor_states, leader_state, end_state, offset_state, error_state])
                observations[drone_id] = obs.astype(np.float32)
            
            elif self.task_type == 'encirclement':
                # 包围任务观测
                # 目标位置（归一化）
                target_state = np.array([
                    self.target_position[0] / self.space_size[0],
                    self.target_position[1] / self.space_size[1],
                    self.target_position[2] / self.space_size[2]
                ])
                
                # 目标速度（归一化）
                target_velocity_state = np.array([
                    (self.target_velocity[0] + 1.0) / 2.0,
                    (self.target_velocity[1] + 1.0) / 2.0,
                    (self.target_velocity[2] + 1.0) / 2.0
                ])
                
                # 其他无人机的包围状态（是否在包围半径内）
                other_encirclement_states = []
                for j in range(self.num_drones):
                    if j != i:
                        distance = np.linalg.norm(self.drone_positions[j] - self.target_position)
                        is_in_radius = 1.0 if distance < self.encirclement_radius else 0.0
                        other_encirclement_states.append(is_in_radius)
                
                # 如果没有其他无人机，填充0
                while len(other_encirclement_states) < 3:
                    other_encirclement_states.append(0.0)
                
                # 包围时间（归一化）
                encirclement_state = np.array([self.encirclement_time / 50.0])
                
                # 组合观测
                obs = np.concatenate([local_state, neighbor_states, target_state, target_velocity_state, other_encirclement_states, encirclement_state])
                observations[drone_id] = obs.astype(np.float32)
        
        return observations
    
    def render(self):
        """渲染环境状态（用于调试）"""
        env_logger.debug(f"步骤 {self.current_step}:")
        for i in range(self.num_drones):
            pos = self.drone_positions[i]
            vel = self.drone_velocities[i]
            battery = self.drone_batteries[i]
            env_logger.debug(f"  无人机{i}: 位置={pos}, 速度={vel}, 电量={battery:.2f}")
        
        if self.task_type == 'inspection':
            env_logger.debug(f"  巡检进度: {self.waypoints_visited}/{self.num_waypoints}")
        elif self.task_type == 'formation':
            env_logger.debug(f"  队形误差: {self.formation_error:.2f}")
    
    def set_task_type(self, task_type: str):
        """
        设置任务类型
        
        参数:
            task_type: 'inspection' 或 'formation'
        """
        self.task_type = task_type
        env_logger.info(f"任务类型设置为: {task_type}")
    
    def set_formation_type(self, formation_type: str):
        """
        设置队形类型
        
        参数:
            formation_type: 'triangle', 'v_shape' 或 'line'
        """
        self.formation_type = formation_type
        env_logger.info(f"队形类型设置为: {formation_type}")
