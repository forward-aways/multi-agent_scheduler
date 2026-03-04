"""
多智能体任务调度系统物流可视化组件
实现物流状态和配送调度的高级可视化
"""

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QProgressBar, QTableWidget, QTableWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem, QTreeWidget, QTreeWidgetItem,
    QSlider, QSpinBox, QCheckBox, QComboBox, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPen, QFont, QPainter
import random
from utils.logging_config import ProjectLogger

logistics_viz_logger = ProjectLogger('logistics_viz', log_dir='logs')


class ZoomableGraphicsView(QGraphicsView):
    """支持缩放的图形视图"""
    
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale_factor = 1.15  # 缩放因子
        
    def wheelEvent(self, event):
        """处理滚轮事件，支持Ctrl+滚轮缩放"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl键被按下，执行缩放
            if event.angleDelta().y() > 0:
                # 向上滚动，放大
                self.scale(self.scale_factor, self.scale_factor)
            else:
                # 向下滚动，缩小
                self.scale(1 / self.scale_factor, 1 / self.scale_factor)
            event.accept()
        else:
            # 没有按下Ctrl键，使用默认滚动行为
            super().wheelEvent(event)


class LogisticsVisualizationWidget(QWidget):
    """物流调度可视化组件"""
    
    def __init__(self):
        super().__init__()
        self.nodes_data = [
            {
                'id': i, 
                'inventory': random.randint(80, 120), 
                'orders': random.randint(0, 5),
                'vehicles': random.randint(2, 5),
                'status': 'normal'
            }
            for i in range(3)
        ]
        self.orders_data = []
        self.deliveries_data = []
        self.local_orders = []  # 本地订单列表
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标题
        title = QLabel("物流调度实时监控")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 创建任务添加面板
        task_panel = QGroupBox("添加物流订单")
        task_layout = QHBoxLayout(task_panel)
        
        # 订单类型选择
        task_layout.addWidget(QLabel("订单类型:"))
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["普通订单", "加急订单", "批量订单"])
        task_layout.addWidget(self.order_type_combo)
        
        # 优先级选择
        task_layout.addWidget(QLabel("优先级:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["高", "中", "低"])
        task_layout.addWidget(self.priority_combo)
        
        # 商品数量
        task_layout.addWidget(QLabel("商品数量:"))
        self.quantity_spinbox = QSpinBox()
        self.quantity_spinbox.setRange(1, 100)
        self.quantity_spinbox.setValue(10)
        task_layout.addWidget(self.quantity_spinbox)
        
        # 添加订单按钮
        add_order_btn = QPushButton("添加订单")
        add_order_btn.clicked.connect(self.add_order)
        task_layout.addWidget(add_order_btn)
        
        layout.addWidget(task_panel)
        
        # 创建控制面板
        control_group = QGroupBox("系统控制")
        control_layout = QHBoxLayout(control_group)
        
        # 添加控制元素
        schedule_btn = QLabel("调度配送")
        schedule_btn.setStyleSheet("background-color: lightblue; padding: 5px; border: 1px solid gray;")
        
        control_layout.addWidget(schedule_btn)
        control_layout.addStretch()
        
        layout.addWidget(control_group)
        
        # 创建主分割布局
        main_split = QHBoxLayout()
        
        # 左侧：物流节点状态
        nodes_group = QGroupBox("物流节点状态")
        nodes_layout = QVBoxLayout(nodes_group)
        
        # 创建节点状态表格
        self.nodes_table = QTableWidget(3, 5)
        self.nodes_table.setHorizontalHeaderLabels(['节点ID', '总库存', '订单数', '车辆数', '状态'])
        self.nodes_table.setShowGrid(True)  # 显示网格线
        self.nodes_table.setGridStyle(Qt.PenStyle.SolidLine)  # 实线网格
        self.nodes_table.setStyleSheet("QTableWidget { gridline-color: lightgray; }")  # 浅灰色网格线
        
        for i in range(3):
            self.update_node_row(i)
        
        nodes_layout.addWidget(self.nodes_table)
        main_split.addWidget(nodes_group)
        
        # 右侧：可视化图表
        viz_group = QGroupBox("物流网络图")
        viz_layout = QVBoxLayout(viz_group)
        
        # 创建图形场景和视图（使用支持缩放的视图）
        self.scene = QGraphicsScene()
        self.graphics_view = ZoomableGraphicsView(self.scene)
        
        viz_layout.addWidget(self.graphics_view)
        main_split.addWidget(viz_group)
        
        layout.addLayout(main_split)
        
        # 下半部分：订单和配送信息
        bottom_split = QHBoxLayout()
        
        # 订单信息
        orders_group = QGroupBox("订单管理")
        orders_layout = QVBoxLayout(orders_group)
        
        self.orders_table = QTableWidget(0, 4)
        self.orders_table.setHorizontalHeaderLabels(['订单ID', '状态', '数量', '目的地'])
        self.orders_table.setShowGrid(True)  # 显示网格线
        self.orders_table.setGridStyle(Qt.PenStyle.SolidLine)  # 实线网格
        self.orders_table.setStyleSheet("QTableWidget { gridline-color: lightgray; }")  # 浅灰色网格线
        orders_layout.addWidget(self.orders_table)
        
        bottom_split.addWidget(orders_group)
        
        # 配送信息
        deliveries_group = QGroupBox("配送管理")
        deliveries_layout = QVBoxLayout(deliveries_group)
        
        self.deliveries_table = QTableWidget(0, 4)
        self.deliveries_table.setHorizontalHeaderLabels(['配送ID', '状态', '路线', '预计时间'])
        self.deliveries_table.setShowGrid(True)  # 显示网格线
        self.deliveries_table.setGridStyle(Qt.PenStyle.SolidLine)  # 实线网格
        self.deliveries_table.setStyleSheet("QTableWidget { gridline-color: lightgray; }")  # 浅灰色网格线
        deliveries_layout.addWidget(self.deliveries_table)
        
        bottom_split.addWidget(deliveries_group)
        
        layout.addLayout(bottom_split)
        
    def setup_timers(self):
        """设置定时器以更新可视化"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(2000)  # 每2秒更新一次
        
    def add_order(self):
        """手动添加物流订单"""
        # 获取订单参数
        order_type = self.order_type_combo.currentText()
        priority_text = self.priority_combo.currentText()
        quantity = self.quantity_spinbox.value()
        
        # 将中文优先级转换为数值
        priority_map = {"高": 5, "中": 3, "低": 1}
        priority_value = priority_map[priority_text]
        
        # 创建订单对象
        order = {
            'type': order_type,
            'priority': priority_value,
            'quantity': quantity,
            'origin': random.choice(['N0', 'N1', 'N2']),  # 随机起始节点
            'destination': random.choice(['N0', 'N1', 'N2'])  # 随机目标节点
        }
        
        # 发送到后端控制器
        logistics_viz_logger.info(f"准备发送订单到后端: {order}")
        logistics_viz_logger.info(f"hasattr(self, 'backend_controller'): {hasattr(self, 'backend_controller')}")
        logistics_viz_logger.info(f"self.backend_controller: {getattr(self, 'backend_controller', None)}")
        
        if hasattr(self, 'backend_controller') and self.backend_controller:
            logistics_viz_logger.info("调用 backend_controller.add_task_to_env")
            self.backend_controller.add_task_to_env(order)
        else:
            logistics_viz_logger.info(f"添加物流订单: 类型={order_type}, 优先级={priority_text}, 数量={quantity}")
        
        # 添加到订单数据列表用于显示
        order_id = f"ORDER_{len(self.orders_data) + 1:03d}"
        self.orders_data.append({
            'id': order_id,
            'type': order_type,
            'priority': priority_value,
            'quantity': quantity,
            'destination': order['destination'],
            'status': 'pending'
        })
        
        # 更新订单表格
        self.orders_table.setRowCount(self.orders_table.rowCount() + 1)
        row = self.orders_table.rowCount() - 1
        self.orders_table.setItem(row, 0, QTableWidgetItem(order_id))
        
        status_item = QTableWidgetItem('待处理')
        status_item.setBackground(QColor(255, 255, 200))  # 浅黄色
        self.orders_table.setItem(row, 1, status_item)
        self.orders_table.setItem(row, 2, QTableWidgetItem(str(quantity)))
        self.orders_table.setItem(row, 3, QTableWidgetItem(order['destination']))
    
    def update_from_backend(self, data):
        """从后端更新数据"""
        # 保存原始数据供可视化使用
        self.backend_data = data
        
        if data:  # 如果有数据，更新显示
            # 解析后端发送的数据
            warehouses = []
            vehicles = []
            orders = []
            stats = None
            
            for item in data:
                if item.get('type') == 'warehouse':
                    warehouses.append(item)
                elif item.get('type') == 'vehicle':
                    vehicles.append(item)
                elif item.get('type') == 'order':
                    orders.append(item)
                elif item.get('type') == 'stats':
                    stats = item
            
            # 更新节点数据
            self.nodes_data = []
            for warehouse in warehouses:
                self.nodes_data.append({
                    'id': warehouse['id'],
                    'position': warehouse['position'],
                    'inventory': warehouse['inventory'],
                    'orders': warehouse['orders'],
                    'vehicles': len([v for v in vehicles if v['target_warehouse'] == warehouse['id']]),
                    'status': warehouse['status']
                })
            
            # 更新订单数据 - 直接使用后端发送的订单状态
            self.orders_data = []
            for order in orders:
                # 状态映射：pending->待处理, delivering->配送中, completed->已完成, failed->失败
                status_map = {
                    'pending': '待处理',
                    'delivering': '配送中',
                    'completed': '已完成',
                    'failed': '失败'
                }
                
                self.orders_data.append({
                    'id': f"ORDER_{order['id'] + 1:03d}",
                    'quantity': order['quantity'],
                    'priority': order['priority'],
                    'destination': f"({order['position'][0]:.1f}, {order['position'][1]:.1f})",
                    'status': status_map.get(order.get('status', 'pending'), '待处理')
                })
            
            # 更新配送数据（从车辆状态推断）
            self.deliveries_data = []
            for vehicle in vehicles:
                if vehicle['status'] == 2:  # 配送中
                    self.deliveries_data.append({
                        'id': vehicle['id'],
                        'status': 'delivering',
                        'route': f"车辆{vehicle['id']} -> 订单{vehicle['target_order']}",
                        'estimated_time': '10分钟'
                    })
            
            # 更新统计信息
            if stats:
                self.stats = stats
            
            self.update_tables()
            self.update_visualization()
        else:  # 如果数据为空，重置到初始状态
            # 重置物流节点数据为初始状态
            self.nodes_data = [
                {
                    'id': i,
                    'position': [20 + i * 30, 20 + i * 30],
                    'inventory': 50,
                    'orders': 0,
                    'vehicles': 0,
                    'status': 'normal'
                } for i in range(3)
            ]
            # 清空所有数据列表
            self.orders_data = []
            self.deliveries_data = []
            # 清空相关表格
            if hasattr(self, 'orders_table') and self.orders_table:
                self.orders_table.setRowCount(0)
            if hasattr(self, 'deliveries_table') and self.deliveries_table:
                self.deliveries_table.setRowCount(0)
            self.update_tables()
            self.update_visualization()
    
    def update_tables(self):
        """更新表格显示"""
        for i in range(min(len(self.nodes_data), 3)):
            self.update_node_row(i)
            
        # 更新订单信息
        self.orders_table.setRowCount(0)
        for order in self.orders_data:
            row = self.orders_table.rowCount()
            self.orders_table.insertRow(row)
            self.orders_table.setItem(row, 0, QTableWidgetItem(str(order.get('id', ''))))
            
            # 将状态转换为中文并设置颜色
            status = order.get('status', '')
            status_map = {
                'pending': '待处理',
                'delivering': '配送中',
                'completed': '已完成',
                'failed': '失败'
            }
            status_display = status_map.get(status, status)
            status_item = QTableWidgetItem(status_display)
            
            # 根据状态设置背景颜色（待处理不设置颜色）
            if status == 'delivering':
                status_item.setBackground(QColor(200, 200, 255))  # 浅蓝色
            elif status == 'completed':
                status_item.setBackground(QColor(200, 255, 200))  # 浅绿色
            elif status == 'failed':
                status_item.setBackground(QColor(255, 200, 200))  # 浅红色
            
            self.orders_table.setItem(row, 1, status_item)
            self.orders_table.setItem(row, 2, QTableWidgetItem(str(order.get('quantity', 0))))
            self.orders_table.setItem(row, 3, QTableWidgetItem(order.get('destination', '')))
        
        # 更新配送信息
        self.deliveries_table.setRowCount(0)
        for delivery in self.deliveries_data:
            row = self.deliveries_table.rowCount()
            self.deliveries_table.insertRow(row)
            self.deliveries_table.setItem(row, 0, QTableWidgetItem(str(delivery.get('id', ''))))
            self.deliveries_table.setItem(row, 1, QTableWidgetItem(delivery.get('status', '')))
            self.deliveries_table.setItem(row, 2, QTableWidgetItem(delivery.get('route', '')))
            self.deliveries_table.setItem(row, 3, QTableWidgetItem(delivery.get('estimated_time', '')))
    
    def update_display(self, data):
        """更新显示（兼容方法名）"""
        self.update_from_backend(data)
        
    def update_node_row(self, row):
        """更新物流节点状态表格中的行"""
        node = self.nodes_data[row]
        
        # 节点ID
        self.nodes_table.setItem(row, 0, QTableWidgetItem(f"Node-{node.get('id', row)}"))
        
        # 库存
        inventory = node.get('inventory', 0)
        inv_text = f"{inventory:.0f}"
        self.nodes_table.setItem(row, 1, QTableWidgetItem(inv_text))
        
        # 订单数
        self.nodes_table.setItem(row, 2, QTableWidgetItem(str(node.get('orders', 0))))
        
        # 车辆数
        vehicle_count = node.get('vehicles', 0)
        vehicle_item = QTableWidgetItem(str(vehicle_count))
        vehicle_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nodes_table.setItem(row, 3, vehicle_item)
        
        # 状态
        status = node.get('status', 'normal')
        status_item = QTableWidgetItem(status)
        if status == 'normal':
            status_item.setBackground(QColor(100, 255, 100))  # 绿色表示正常
        elif status == 'warning':
            status_item.setBackground(QColor(255, 255, 100))  # 黄色表示警告
        else:
            status_item.setBackground(QColor(255, 100, 100))  # 红色表示异常
        self.nodes_table.setItem(row, 4, status_item)
    
    def update_visualization(self):
        """更新可视化图表"""
        # 清除现有场景
        self.scene.clear()
        
        # 绘制物流节点（仓库）
        for i, node in enumerate(self.nodes_data):
            # 获取仓库位置
            pos = node.get('position', [20 + i * 30, 20 + i * 30])
            x, y = pos[0] * 5, pos[1] * 5  # 缩放位置以适应视图
            
            # 绘制节点圆圈
            color = QColor(100, 200, 100)  # 默认绿色
            status = node.get('status', 'normal')
            if status == 'warning':
                color = QColor(255, 255, 100)  # 黄色
            elif status == 'error':
                color = QColor(255, 100, 100)  # 红色
            
            node_ellipse = self.scene.addEllipse(x-30, y-30, 60, 60, 
                                                QPen(color), QBrush(color))
            
            # 添加节点标签
            inventory = node.get('inventory', 50)
            vehicles = node.get('vehicles', 0)
            orders = node.get('orders', 0)
            node_text = self.scene.addText(f"仓库{i}\n库存:{inventory}\n订单:{orders}\n车:{vehicles}")
            node_text.setPos(x-25, y-10)
            node_text.setDefaultTextColor(QColor(0, 0, 0))
        
        # 绘制车辆
        # 从后端数据中获取车辆信息
        vehicle_data = []
        if hasattr(self, 'backend_data'):
            for item in self.backend_data:
                if item.get('type') == 'vehicle':
                    vehicle_data.append(item)
        
        # 绘制每辆车
        for vehicle in vehicle_data:
            pos = vehicle.get('position', [0, 0])
            x, y = pos[0] * 5, pos[1] * 5  # 缩放位置
            
            # 根据车辆状态设置颜色
            status = vehicle.get('status', 0)
            if status == 0:  # 空闲
                color = QColor(100, 100, 255)  # 蓝色
            elif status == 1:  # 前往仓库
                color = QColor(255, 255, 100)  # 黄色
            elif status == 2:  # 配送中
                color = QColor(255, 100, 100)  # 红色
            else:
                color = QColor(200, 200, 200)  # 灰色
            
            # 绘制车辆（小矩形）
            vehicle_rect = self.scene.addRect(x-10, y-10, 20, 20,
                                             QPen(color), QBrush(color))
            
            # 添加车辆标签
            cargo = vehicle.get('cargo', 0)
            vehicle_text = self.scene.addText(f"V{vehicle['id']}\n{cargo:.0f}")
            vehicle_text.setPos(x-15, y-25)
            vehicle_text.setDefaultTextColor(QColor(0, 0, 0))
        
        # 绘制订单
        if hasattr(self, 'orders_data'):
            for order in self.orders_data:
                # 解析订单位置
                dest_str = order.get('destination', '')
                if dest_str.startswith('('):
                    try:
                        # 从字符串中提取坐标
                        coords = dest_str.strip('()').split(',')
                        x, y = float(coords[0]), float(coords[1])
                        x, y = x * 5, y * 5  # 缩放位置
                        
                        # 绘制订单标记
                        order_ellipse = self.scene.addEllipse(x-15, y-15, 30, 30, 
                                                           QPen(QColor(255, 0, 0)), QBrush(QColor(255, 200, 200)))
                        
                        # 添加订单标签
                        order_text = self.scene.addText(f"订单{order['id']}\n数量:{order['quantity']}\n优先级:{order['priority']}")
                        order_text.setPos(x-20, y-20)
                        order_text.setDefaultTextColor(QColor(0, 0, 0))
                    except:
                        pass
        
        # 绘制连接线（表示物流路线）
        for i in range(len(self.nodes_data)):
            for j in range(i+1, len(self.nodes_data)):
                pos1 = self.nodes_data[i].get('position', [20 + i * 30, 20 + i * 30])
                pos2 = self.nodes_data[j].get('position', [20 + j * 30, 20 + j * 30])
                x1, y1 = pos1[0] * 5, pos1[1] * 5
                x2, y2 = pos2[0] * 5, pos2[1] * 5
                
                # 绘制连接线
                line = self.scene.addLine(x1, y1, x2, y2, QPen(QColor(150, 150, 150), 2))
                
                # 添加流量信息
                flow_text = self.scene.addText(">>")
                flow_text.setPos((x1+x2)//2, (y1+y2)//2)
                flow_text.setDefaultTextColor(QColor(0, 0, 255))