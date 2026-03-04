"""
多智能体物流调度环境
用于训练多智能体协作的物流调度系统
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, List
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import ProjectLogger

env_logger = ProjectLogger('logistics_env', log_dir='logs')


class MultiAgentLogisticsEnv(gym.Env):
    """
    多智能体物流调度环境
    包括仓库和车辆两种智能体，协同完成配送任务
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境
        
        参数:
            config: 环境配置参数
        """
        super().__init__()
        
        # 仓库配置
        self.num_warehouses = config.get('num_warehouses', 3)
        self.warehouse_capacity = config.get('warehouse_capacity', 100)
        self.warehouse_positions = config.get('warehouse_positions', [
            [20.0, 20.0],
            [50.0, 50.0],
            [80.0, 80.0]
        ])
        
        # 车辆配置
        self.num_vehicles = config.get('num_vehicles', 5)
        self.vehicle_capacity = config.get('vehicle_capacity', 20)
        self.vehicle_speed = config.get('vehicle_speed', 5.0)
        self.vehicle_positions = config.get('vehicle_positions', [
            [20.0, 20.0],
            [30.0, 30.0],
            [50.0, 50.0],
            [70.0, 70.0],
            [80.0, 80.0]
        ])
        
        # 订单配置
        self.order_generation_rate = config.get('order_generation_rate', 2)
        self.max_pending_orders = config.get('max_pending_orders', 15)
        self.map_size = config.get('map_size', [100.0, 100.0])
        
        # 手动模式标志（True时不自动生成订单）
        self.manual_mode = config.get('manual_mode', False)
        
        # 时间配置
        self.max_steps = config.get('max_steps', 200)
        
        # 仓库状态
        self.warehouse_inventory = np.zeros(self.num_warehouses)  # 每个仓库的库存
        self.warehouse_orders = [[] for _ in range(self.num_warehouses)]  # 每个仓库的订单
        
        # 车辆状态
        self.vehicle_positions = np.array(self.vehicle_positions)
        self.vehicle_cargo = np.zeros(self.num_vehicles)  # 每个车辆的载货量
        self.vehicle_status = np.zeros(self.num_vehicles)  # 0: 空闲, 1: 去仓库, 2: 去配送, 3: 返回仓库
        self.vehicle_target_warehouse = np.zeros(self.num_vehicles, dtype=int)  # 车辆目标仓库
        self.vehicle_target_order_pos = [None] * self.num_vehicles  # 车辆目标订单位置
        
        # 订单队列
        self.pending_orders = []  # 待处理订单 [位置, 数量, 优先级]
        self.delivering_orders = []  # 配送中的订单 [位置, 数量, 优先级, 是否已分配]
        self.completed_orders = 0
        self.failed_orders = 0
        self.all_orders = []  # 所有订单列表，包括状态 [{'position': [...], 'quantity': ..., 'priority': ..., 'status': 'pending/completed/failed', 'assigned_warehouse': ...}]
        self.warehouse_rejected_orders = [0] * self.num_warehouses  # 每个仓库拒绝的订单数量
        self.last_completed_orders = 0  # 上一步完成的订单数
        self.last_failed_orders = 0  # 上一步失败的订单数
        
        # 时间跟踪
        self.current_step = 0
        
        # 为每个智能体定义动作空间和观测空间
        self.action_spaces = {}
        self.observation_spaces = {}
        
        # 仓库智能体
        for i in range(self.num_warehouses):
            # 仓库动作空间：分配订单(0), 拒绝订单(1), 调整库存(2)
            self.action_spaces[f'warehouse_{i}'] = gym.spaces.Discrete(3)
            
            # 仓库观测空间
            # [库存量, 订单数, 邻居仓库状态, 待处理订单信息]
            obs_dim = 1 + 1 + (self.num_warehouses - 1) * 2 + min(5, self.max_pending_orders) * 3
            env_logger.debug(f"仓库{i}观测维度: 本地状态2 + 邻居{(self.num_warehouses-1)*2} + 订单{min(5, self.max_pending_orders)*3} = {obs_dim}")
            self.observation_spaces[f'warehouse_{i}'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
        
        # 车辆智能体
        for i in range(self.num_vehicles):
            # 车辆动作空间：去仓库(0), 去配送(1), 返回仓库(2), 等待(3)
            self.action_spaces[f'vehicle_{i}'] = gym.spaces.Discrete(4)
            
            # 车辆观测空间
            # [位置, 载货量, 状态, 目标仓库, 目标订单, 邻居车辆状态, 待处理订单信息]
            obs_dim = 2 + 1 + 1 + 1 + 1 + (self.num_vehicles - 1) * 3 + min(5, self.max_pending_orders) * 3
            env_logger.debug(f"车辆{i}观测维度: 本地状态6 + 邻居{(self.num_vehicles-1)*3} + 订单{min(5, self.max_pending_orders)*3} = {obs_dim}")
            self.observation_spaces[f'vehicle_{i}'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        返回:
            初始观测和信息
        """
        super().reset(seed=seed)
        
        # 重置仓库状态
        self.warehouse_inventory = np.ones(self.num_warehouses) * (self.warehouse_capacity * 0.5)
        self.warehouse_orders = [[] for _ in range(self.num_warehouses)]
        
        # 重置车辆状态 - 使用原始配置中的位置
        original_positions = [
            [20.0, 20.0],
            [30.0, 30.0],
            [50.0, 50.0],
            [70.0, 70.0],
            [80.0, 80.0]
        ]
        self.vehicle_positions = np.array(original_positions[:self.num_vehicles])
        self.vehicle_cargo = np.zeros(self.num_vehicles)
        self.vehicle_status = np.zeros(self.num_vehicles)
        self.vehicle_target_warehouse = np.zeros(self.num_vehicles, dtype=int)
        self.vehicle_target_order_pos = [None] * self.num_vehicles
        
        # 重置订单队列
        self.pending_orders = []
        self.delivering_orders = []
        self.completed_orders = 0
        self.failed_orders = 0
        self.all_orders = []  # 清空所有订单列表
        self.warehouse_rejected_orders = [0] * self.num_warehouses  # 重置每个仓库拒绝的订单数量
        self.last_completed_orders = 0  # 重置上一步完成的订单数
        self.last_failed_orders = 0  # 重置上一步失败的订单数
        
        # 重置时间
        self.current_step = 0
        
        # 生成初始订单
        self._generate_orders()
        
        env_logger.info(f"物流调度环境重置，仓库数: {self.num_warehouses}, 车辆数: {self.num_vehicles}")
        
        return self._get_observations(), {}
    
    def step(self, actions: Dict[str, int]):
        """
        执行一步
        
        参数:
            actions: 每个智能体的动作
            
        返回:
            下一步观测, 奖励, 是否结束, 是否截断, 信息
        """
        self.current_step += 1
        
        try:
            # 执行仓库动作
            for i in range(self.num_warehouses):
                warehouse_id = f'warehouse_{i}'
                action = actions.get(warehouse_id, 0)
                self._execute_warehouse_action(i, action)
        except Exception as e:
            env_logger.error(f"执行仓库动作时出错: {str(e)}")
            raise
        
        try:
            # 执行车辆动作
            for i in range(self.num_vehicles):
                vehicle_id = f'vehicle_{i}'
                action = actions.get(vehicle_id, 0)
                self._execute_vehicle_action(i, action)
        except Exception as e:
            env_logger.error(f"执行车辆动作时出错: {str(e)}")
            raise
        
        try:
            # 更新车辆位置
            self._update_vehicle_positions()
        except Exception as e:
            env_logger.error(f"更新车辆位置时出错: {str(e)}")
            raise
        
        try:
            # 生成新订单
            self._generate_orders()
        except Exception as e:
            env_logger.error(f"生成新订单时出错: {str(e)}")
            raise
        
        try:
            # 检查订单状态
            self._check_order_status()
        except Exception as e:
            env_logger.error(f"检查订单状态时出错: {str(e)}")
            raise
        
        try:
            # 计算奖励
            rewards = self._compute_rewards()
        except Exception as e:
            env_logger.error(f"计算奖励时出错: {str(e)}")
            raise
        
        # 检查是否结束
        terminated = {f'warehouse_{i}': False for i in range(self.num_warehouses)}
        terminated.update({f'vehicle_{i}': False for i in range(self.num_vehicles)})
        
        truncated = {f'warehouse_{i}': False for i in range(self.num_warehouses)}
        truncated.update({f'vehicle_{i}': False for i in range(self.num_vehicles)})
        
        if self.current_step >= self.max_steps:
            for key in terminated:
                terminated[key] = True
                truncated[key] = True
        
        try:
            # 获取观测
            observations = self._get_observations()
        except Exception as e:
            env_logger.error(f"获取观测时出错: {str(e)}")
            raise
        
        # 获取信息
        infos = self._get_info()
        
        return observations, rewards, terminated, truncated, infos
    
    def _execute_warehouse_action(self, warehouse_idx: int, action: int):
        """执行仓库动作"""
        if action == 0:  # 分配订单
            if len(self.pending_orders) > 0 and self.warehouse_inventory[warehouse_idx] > 0:
                # 选择优先级最高的订单
                if len(self.pending_orders) > 0:
                    order_idx = 0
                    max_priority = float(self.pending_orders[0][2])
                    for i, order in enumerate(self.pending_orders[:5]):
                        if float(order[2]) > max_priority:
                            max_priority = float(order[2])
                            order_idx = i
                    order = self.pending_orders[order_idx]
                else:
                    order = None
                
                # 检查库存是否足够
                if order is not None and float(order[1]) <= self.warehouse_inventory[warehouse_idx]:
                    self.warehouse_inventory[warehouse_idx] -= float(order[1])
                    self.warehouse_orders[warehouse_idx].append(order)
                    
                    # 将订单从pending_orders移到delivering_orders
                    # 添加第4个元素标记是否已分配给车辆（False=未分配）
                    self.pending_orders.pop(order_idx)
                    order_with_assigned_flag = [order[0], order[1], order[2], False]
                    self.delivering_orders.append(order_with_assigned_flag)
                    
                    # 更新all_orders中的订单状态为"配送中"
                    for order_info in self.all_orders:
                        if (np.array_equal(order_info['position'], order[0]) and 
                            order_info['quantity'] == order[1] and 
                            order_info['priority'] == order[2] and
                            order_info['status'] == 'pending'):
                            order_info['status'] = 'delivering'
                            order_info['assigned_warehouse'] = warehouse_idx
                            env_logger.info(f"订单状态更新: 配送中 - 位置={order[0]}, 数量={order[1]}, 优先级={order[2]}, 分配给仓库{warehouse_idx}")
                            break
                    
                    env_logger.debug(f"仓库{warehouse_idx}分配订单: {order_with_assigned_flag}")
        
        elif action == 1:  # 拒绝订单
            if len(self.pending_orders) > 0:
                # 选择优先级最低的订单拒绝
                order_idx = 0
                min_priority = float(self.pending_orders[0][2])
                for i, order in enumerate(self.pending_orders[:5]):
                    if float(order[2]) < min_priority:
                        min_priority = float(order[2])
                        order_idx = i
                order = self.pending_orders[order_idx]
                self.pending_orders.pop(order_idx)
                self.failed_orders += 1
                self.warehouse_rejected_orders[warehouse_idx] += 1  # 记录该仓库拒绝的订单数量
                
                # 更新all_orders中的订单状态为"失败"
                for order_info in self.all_orders:
                    if (np.array_equal(order_info['position'], order[0]) and 
                        order_info['quantity'] == order[1] and 
                        order_info['priority'] == order[2] and
                        order_info['status'] == 'pending'):
                        order_info['status'] = 'failed'
                        order_info['assigned_warehouse'] = warehouse_idx
                        break
                
                env_logger.debug(f"仓库{warehouse_idx}拒绝订单: {order}，给予惩罚")
        
        elif action == 2:  # 调整库存
            # 简单的库存调整：补充库存
            if self.warehouse_inventory[warehouse_idx] < self.warehouse_capacity * 0.3:
                self.warehouse_inventory[warehouse_idx] += 10
                env_logger.debug(f"仓库{warehouse_idx}补充库存: {self.warehouse_inventory[warehouse_idx]}")
    
    def _execute_vehicle_action(self, vehicle_idx: int, action: int):
        """执行车辆动作"""
        env_logger.debug(f"车辆{vehicle_idx}执行动作{action}, 当前状态={self.vehicle_status[vehicle_idx]}, 载货={self.vehicle_cargo[vehicle_idx]}")
        
        if action == 0:  # 去仓库
            if self.vehicle_status[vehicle_idx] == 0:  # 空闲状态
                # 选择有订单的仓库中最近的一个
                warehouses_with_orders = [i for i in range(self.num_warehouses) 
                                         if len(self.warehouse_orders[i]) > 0]
                
                if len(warehouses_with_orders) > 0:
                    # 有订单的仓库中选择最近的
                    distances = [np.linalg.norm(self.vehicle_positions[vehicle_idx] - 
                                               np.array(self.warehouse_positions[i])) 
                               for i in warehouses_with_orders]
                    nearest_idx = warehouses_with_orders[int(np.argmin(distances))]
                else:
                    # 没有仓库有订单，选择最近的仓库
                    distances = [np.linalg.norm(self.vehicle_positions[vehicle_idx] - np.array(pos)) 
                               for pos in self.warehouse_positions]
                    nearest_idx = int(np.argmin(distances))
                
                self.vehicle_target_warehouse[vehicle_idx] = nearest_idx
                self.vehicle_status[vehicle_idx] = 1  # 去仓库
                env_logger.debug(f"车辆{vehicle_idx}去仓库{self.vehicle_target_warehouse[vehicle_idx]}")
        
        elif action == 1:  # 装货后去配送
            env_logger.debug(f"车辆{vehicle_idx}尝试去配送：状态={self.vehicle_status[vehicle_idx]}, 载货={self.vehicle_cargo[vehicle_idx]}, 配送中订单数={len(self.delivering_orders)}")
            if self.vehicle_status[vehicle_idx] == 0 and self.vehicle_cargo[vehicle_idx] > 0:
                # 选择最近的未分配配送中订单
                unassigned_orders = [(i, order) for i, order in enumerate(self.delivering_orders[:5]) if len(order) < 4 or not order[3]]
                
                if len(unassigned_orders) > 0:
                    distances = [np.linalg.norm(self.vehicle_positions[vehicle_idx] - np.array(order[0])) 
                               for _, order in unassigned_orders]
                    min_idx = int(np.argmin(distances))
                    order_idx, selected_order = unassigned_orders[min_idx]
                    
                    # 标记订单为已分配
                    if len(selected_order) < 4:
                        selected_order.append(True)
                    else:
                        selected_order[3] = True
                    
                    self.vehicle_target_order_pos[vehicle_idx] = selected_order[0].copy()
                    self.vehicle_status[vehicle_idx] = 2  # 去配送
                    env_logger.debug(f"车辆{vehicle_idx}去配送订单{order_idx}, 位置={self.vehicle_target_order_pos[vehicle_idx]}")
                else:
                    env_logger.debug(f"车辆{vehicle_idx}没有未分配的配送中订单，无法去配送")
            else:
                env_logger.debug(f"车辆{vehicle_idx}不满足去配送条件：状态={self.vehicle_status[vehicle_idx]}, 载货={self.vehicle_cargo[vehicle_idx]}")
        
        elif action == 2:  # 空载时返回仓库（可选动作）
            if self.vehicle_status[vehicle_idx] == 0 and self.vehicle_cargo[vehicle_idx] == 0:
                # 选择最近的仓库返回
                distances = [np.linalg.norm(self.vehicle_positions[vehicle_idx] - np.array(pos)) 
                           for pos in self.warehouse_positions]
                self.vehicle_target_warehouse[vehicle_idx] = int(np.argmin(distances))
                self.vehicle_status[vehicle_idx] = 3  # 返回仓库
                env_logger.debug(f"车辆{vehicle_idx}返回仓库{self.vehicle_target_warehouse[vehicle_idx]}")
        
        elif action == 3:  # 等待/保持当前状态
            pass
    
    def _update_vehicle_positions(self):
        """更新车辆位置"""
        for i in range(self.num_vehicles):
            if self.vehicle_status[i] == 1:  # 去仓库
                # 确保目标仓库索引有效
                target_warehouse_idx = self.vehicle_target_warehouse[i]
                if target_warehouse_idx >= self.num_warehouses:
                    env_logger.warning(f"车辆{i}目标仓库索引{target_warehouse_idx}越界，重置为0")
                    target_warehouse_idx = 0
                    self.vehicle_target_warehouse[i] = 0
                
                target_pos = np.array(self.warehouse_positions[target_warehouse_idx])
                direction = target_pos - self.vehicle_positions[i]
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    self.vehicle_positions[i] += (direction / distance) * self.vehicle_speed
                
                # 检查是否到达仓库
                if distance < self.vehicle_speed or distance == 0:
                    self.vehicle_positions[i] = target_pos
                    # 到达仓库后自动装货
                    if len(self.warehouse_orders[target_warehouse_idx]) > 0:
                        order = self.warehouse_orders[target_warehouse_idx].pop(0)
                        # 检查容量限制，不能超过车辆最大容量
                        order_quantity = float(order[1])
                        if order_quantity > self.vehicle_capacity:
                            env_logger.warning(f"车辆{i}订单数量{order_quantity}超过容量{self.vehicle_capacity}，只装载{self.vehicle_capacity}")
                            self.vehicle_cargo[i] = self.vehicle_capacity
                        else:
                            self.vehicle_cargo[i] = order_quantity
                        # 设置目标订单位置
                        self.vehicle_target_order_pos[i] = order[0].copy()
                        # 装货后直接进入待配送状态（空闲但载货）
                        self.vehicle_status[i] = 0  # 空闲状态，等待模型选择去配送
                        env_logger.debug(f"车辆{i}在仓库装货：载货={self.vehicle_cargo[i]}, 目标订单={self.vehicle_target_order_pos[i]}")
                    else:
                        # 仓库没有订单，保持空闲
                        self.vehicle_status[i] = 0
                        env_logger.debug(f"车辆{i}到达仓库但无订单可装")
            
            elif self.vehicle_status[i] == 2:  # 去配送
                if self.vehicle_target_order_pos[i] is not None:
                    target_pos = np.array(self.vehicle_target_order_pos[i])
                    direction = target_pos - self.vehicle_positions[i]
                    distance = np.linalg.norm(direction)
                    
                    env_logger.debug(f"车辆{i}去配送: 当前位置={self.vehicle_positions[i]}, 目标位置={target_pos}, 距离={distance:.2f}")
                    
                    if distance > 0:
                        self.vehicle_positions[i] += (direction / distance) * self.vehicle_speed
                        env_logger.debug(f"车辆{i}移动后位置: {self.vehicle_positions[i]}")
                    
                    # 检查是否到达配送点
                    if distance <= self.vehicle_speed:
                        self.vehicle_positions[i] = target_pos
                        self.vehicle_cargo[i] = 0  # 卸货
                        self.vehicle_target_order_pos[i] = None  # 清除目标订单
                        
                        # 从 delivering_orders 中移除订单
                        for idx, order in enumerate(self.delivering_orders):
                            if np.array_equal(order[0], target_pos):
                                self.delivering_orders.pop(idx)
                                env_logger.debug(f"从配送中移除订单：位置={target_pos}")
                                break
                        
                        # 更新 all_orders 中的订单状态为"已完成"
                        for order_info in self.all_orders:
                            if (np.array_equal(order_info['position'], target_pos) and 
                                order_info['status'] == 'delivering'):
                                order_info['status'] = 'completed'
                                self.completed_orders += 1
                                env_logger.debug(f"订单已完成：位置={target_pos}")
                                break
                        
                        # 配送完成后变为空闲（空载）
                        self.vehicle_status[i] = 0
                        env_logger.debug(f"车辆{i}完成配送，变为空闲")
            
            elif self.vehicle_status[i] == 3:  # 返回仓库
                # 确保目标仓库索引有效
                target_warehouse_idx = self.vehicle_target_warehouse[i]
                if target_warehouse_idx >= self.num_warehouses:
                    env_logger.warning(f"车辆{i}返回仓库索引{target_warehouse_idx}越界，重置为0")
                    target_warehouse_idx = 0
                    self.vehicle_target_warehouse[i] = 0
                
                target_pos = np.array(self.warehouse_positions[target_warehouse_idx])
                direction = target_pos - self.vehicle_positions[i]
                distance = np.linalg.norm(direction)
                if distance > 0:
                    self.vehicle_positions[i] += (direction / distance) * self.vehicle_speed
                
                # 检查是否到达仓库
                if distance < self.vehicle_speed:
                    self.vehicle_positions[i] = target_pos
                    self.vehicle_status[i] = 0  # 到达仓库，变为空闲
    
    def _generate_orders(self):
        """生成新订单"""
        # 手动模式下不自动生成订单
        if self.manual_mode:
            env_logger.debug("手动模式，跳过自动生成订单")
            return
            
        if len(self.pending_orders) < self.max_pending_orders:
            # 随机生成订单
            num_new_orders = int(np.random.poisson(self.order_generation_rate))
            for _ in range(num_new_orders):
                position = np.random.rand(2) * self.map_size
                quantity = np.random.randint(1, 5)
                priority = np.random.randint(1, 5)
                
                # 添加到 pending_orders
                self.pending_orders.append([position, quantity, priority])
                
                # 同时添加到 all_orders 用于状态追踪
                self.all_orders.append({
                    'position': position.copy(),
                    'quantity': quantity,
                    'priority': priority,
                    'status': 'pending',
                    'assigned_warehouse': None
                })
                
                env_logger.debug(f"生成新订单: 位置={position}, 数量={quantity}, 优先级={priority}")
    
    def _check_order_status(self):
        """检查订单状态"""
        # 手动模式下不自动拒绝订单
        if self.manual_mode:
            env_logger.debug(f"手动模式，跳过订单拒绝检查，当前待处理订单数: {len(self.pending_orders)}")
            return
            
        # 检查是否有订单超时
        current_orders = len(self.pending_orders)
        if current_orders > self.max_pending_orders * 0.8:
            # 拒绝部分订单
            num_to_reject = int(current_orders - self.max_pending_orders * 0.8)
            for _ in range(num_to_reject):
                if len(self.pending_orders) > 0:
                    self.pending_orders.pop(0)
                    self.failed_orders += 1
    
    def _compute_rewards(self):
        """
        计算奖励（简化版 - 参考服务器调度成功的设计）
        
        核心设计原则：
        1. 简单明确：奖励信号清晰，不复杂
        2. 即时反馈：每一步都有明确的正负反馈
        3. 任务导向：围绕完成任务设计奖励
        4. 去耦合：每个智能体只对自己的行为负责
        """
        rewards = {}
        
        # 计算增量
        delta_completed = self.completed_orders - self.last_completed_orders
        delta_failed = self.failed_orders - self.last_failed_orders
        
        # 更新上一步的订单数
        self.last_completed_orders = self.completed_orders
        self.last_failed_orders = self.failed_orders
        
        # ==================== 仓库奖励 ====================
        for i in range(self.num_warehouses):
            warehouse_id = f'warehouse_{i}'
            reward = 0.0
            
            # 1. 完成订单奖励（核心奖励）
            if delta_completed > 0:
                reward += delta_completed * 10.0  # 完成订单给予奖励
            
            # 2. 拒绝订单惩罚
            if delta_failed > 0:
                reward -= delta_failed * 5.0  # 拒绝订单给予惩罚
            
            # 3. 分配订单奖励（即时反馈）
            if len(self.warehouse_orders[i]) > 0:
                reward += 1.0  # 每分配一个订单给予小奖励
            
            # 4. 处理订单奖励（鼓励行动）
            if len(self.pending_orders) > 0 and self.warehouse_inventory[i] > 0:
                reward += 0.5  # 有待处理订单且有库存时，给予行动鼓励
            
            rewards[warehouse_id] = reward
        
        # ==================== 车辆奖励 ====================
        for i in range(self.num_vehicles):
            vehicle_id = f'vehicle_{i}'
            reward = 0.0
            
            # 1. 完成配送奖励（核心奖励）
            if delta_completed > 0:
                reward += delta_completed * 10.0  # 完成配送给予奖励
            
            # 2. 装货奖励（即时反馈）
            if self.vehicle_status[i] == 1 and self.vehicle_cargo[i] > 0:
                reward += 2.0  # 装货给予小奖励
            
            # 3. 配送中奖励（鼓励行动）
            if self.vehicle_status[i] == 2:
                reward += 1.0  # 正在配送给予持续奖励
            
            # 4. 返回仓库奖励（鼓励回库）
            if self.vehicle_status[i] == 3:
                reward += 0.5  # 返回仓库给予小奖励
            
            # 5. 空闲惩罚（鼓励行动）
            if self.vehicle_status[i] == 0:
                if len(self.delivering_orders) > 0:
                    reward -= 2.0  # 有订单但空闲，给予惩罚
                else:
                    reward -= 0.5  # 无订单也空闲，轻微惩罚
            
            # 6. 载货率奖励（鼓励满载）
            if self.vehicle_cargo[i] > 0:
                cargo_ratio = self.vehicle_cargo[i] / self.vehicle_capacity
                reward += cargo_ratio * 1.0  # 载货率越高奖励越高
            
            rewards[vehicle_id] = reward
        
        return rewards
    
    def _get_observations(self):
        """获取观测"""
        observations = {}
        
        try:
            # 仓库观测
            for i in range(self.num_warehouses):
                warehouse_id = f'warehouse_{i}'
                
                # 本地状态
                local_state = np.array([
                    self.warehouse_inventory[i] / self.warehouse_capacity,  # 库存比例
                    len(self.warehouse_orders[i]) / 10.0  # 订单数
                ])
                
                # 邻居仓库状态
                neighbor_states = []
                for j in range(self.num_warehouses):
                    if i != j:
                        neighbor_states.extend([
                            self.warehouse_inventory[j] / self.warehouse_capacity,
                            len(self.warehouse_orders[j]) / 10.0
                        ])
                
                # 待处理订单信息
                order_states = []
                for idx, order in enumerate(self.pending_orders[:5]):
                    try:
                        order_states.extend([
                            float(order[0][0]) / self.map_size[0],  # x位置
                            float(order[0][1]) / self.map_size[1],  # y位置
                            float(order[2]) / 5.0  # 优先级
                        ])
                    except Exception as e:
                        env_logger.error(f"处理仓库观测订单{idx}时出错: {str(e)}")
                        env_logger.error(f"订单内容: {order}")
                        env_logger.error(f"订单类型: {type(order)}")
                        if isinstance(order, (list, tuple)) and len(order) > 0:
                            env_logger.error(f"order[0]类型: {type(order[0])}")
                            env_logger.error(f"order[0]内容: {order[0]}")
                        raise
                # 填充到固定长度
                while len(order_states) < min(5, self.max_pending_orders) * 3:
                    order_states.extend([0.0, 0.0, 0.0])
                
                # 组合观测
                obs = np.concatenate([local_state, neighbor_states, order_states])
                observations[warehouse_id] = obs.astype(np.float32)
                env_logger.debug(f"仓库{warehouse_id}实际观测shape: {obs.shape}, 预期shape: ({self.observation_spaces[warehouse_id].shape[0]},)")
        except Exception as e:
            env_logger.error(f"获取仓库观测时出错: {str(e)}")
            raise
        
        try:
            # 车辆观测
            for i in range(self.num_vehicles):
                vehicle_id = f'vehicle_{i}'
                
                # 本地状态
                has_target_order = 1.0 if self.vehicle_target_order_pos[i] is not None else 0.0
                local_state = np.array([
                    self.vehicle_positions[i][0] / self.map_size[0],  # x位置
                    self.vehicle_positions[i][1] / self.map_size[1],  # y位置
                    self.vehicle_cargo[i] / self.vehicle_capacity,  # 载货量
                    self.vehicle_status[i] / 3.0,  # 状态
                    self.vehicle_target_warehouse[i] / self.num_warehouses,  # 目标仓库
                    has_target_order  # 是否有目标订单
                ])
                
                # 邻居车辆状态
                neighbor_states = []
                for j in range(self.num_vehicles):
                    if i != j:
                        neighbor_states.extend([
                            self.vehicle_positions[j][0] / self.map_size[0],
                            self.vehicle_positions[j][1] / self.map_size[1],
                            self.vehicle_cargo[j] / self.vehicle_capacity
                        ])
                
                # 待处理订单信息
                order_states = []
                for idx, order in enumerate(self.pending_orders[:5]):
                    try:
                        order_states.extend([
                            float(order[0][0]) / self.map_size[0],  # x位置
                            float(order[0][1]) / self.map_size[1],  # y位置
                            float(order[2]) / 5.0  # 优先级
                        ])
                    except Exception as e:
                        env_logger.error(f"处理车辆观测订单{idx}时出错: {str(e)}")
                        env_logger.error(f"订单内容: {order}")
                        env_logger.error(f"订单类型: {type(order)}")
                        if isinstance(order, (list, tuple)) and len(order) > 0:
                            env_logger.error(f"order[0]类型: {type(order[0])}")
                            env_logger.error(f"order[0]内容: {order[0]}")
                        raise
                # 填充到固定长度
                while len(order_states) < min(5, self.max_pending_orders) * 3:
                    order_states.extend([0.0, 0.0, 0.0])
                
                # 组合观测
                obs = np.concatenate([local_state, neighbor_states, order_states])
                observations[vehicle_id] = obs.astype(np.float32)
                env_logger.debug(f"车辆{vehicle_id}实际观测shape: {obs.shape}, 预期shape: ({self.observation_spaces[vehicle_id].shape[0]},)")
        except Exception as e:
            env_logger.error(f"获取车辆观测时出错: {str(e)}")
            raise
        
        return observations
    
    def _get_info(self):
        """获取信息"""
        infos = {}
        
        # 仓库信息
        for i in range(self.num_warehouses):
            warehouse_id = f'warehouse_{i}'
            infos[warehouse_id] = {
                'inventory': self.warehouse_inventory[i],
                'orders': len(self.warehouse_orders[i]),
                'completed_orders': self.completed_orders,
                'failed_orders': self.failed_orders
            }
        
        # 车辆信息
        for i in range(self.num_vehicles):
            vehicle_id = f'vehicle_{i}'
            infos[vehicle_id] = {
                'position': self.vehicle_positions[i],
                'cargo': self.vehicle_cargo[i],
                'status': self.vehicle_status[i],
                'completed_orders': self.completed_orders,
                'failed_orders': self.failed_orders
            }
        
        return infos