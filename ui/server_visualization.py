"""
多智能体任务调度系统服务器可视化组件
实现服务器状态和任务分配的高级可视化
"""

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QProgressBar, QTableWidget, QTableWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem, QComboBox, QSpinBox, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QBrush, QColor, QPen, QFont, QPainter
from PyQt6.QtWidgets import QGraphicsScene
import random
from utils.logging_config import ProjectLogger

viz_logger = ProjectLogger('visualization', log_dir='logs')


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


class ServerVisualizationWidget(QWidget):
    """服务器调度可视化组件"""
    
    def __init__(self):
        super().__init__()
        self.servers_data = [
            {'id': i, 'cpu': 0, 'memory': 0, 'tasks': 0, 'status': 'idle'} 
            for i in range(5)
        ]
        self.tasks_data = []
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标题
        title = QLabel("服务器调度实时监控")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 创建任务添加面板
        task_panel = QGroupBox("添加新任务")
        task_layout = QHBoxLayout(task_panel)
        
        # 任务类型选择
        task_layout.addWidget(QLabel("任务类型:"))
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["计算任务", "存储任务", "网络任务"])
        task_layout.addWidget(self.task_type_combo)
        
        # 优先级选择
        task_layout.addWidget(QLabel("优先级:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["高", "中", "低"])
        task_layout.addWidget(self.priority_combo)
        
        # CPU需求
        task_layout.addWidget(QLabel("CPU需求:"))
        self.cpu_spinbox = QSpinBox()
        self.cpu_spinbox.setRange(1, 100)
        self.cpu_spinbox.setValue(20)
        task_layout.addWidget(self.cpu_spinbox)
        
        # 内存需求
        task_layout.addWidget(QLabel("内存需求:"))
        self.memory_spinbox = QSpinBox()
        self.memory_spinbox.setRange(1, 100)
        self.memory_spinbox.setValue(15)
        task_layout.addWidget(self.memory_spinbox)
        
        # 添加任务按钮
        add_task_btn = QPushButton("添加任务")
        add_task_btn.clicked.connect(self.add_task)
        task_layout.addWidget(add_task_btn)
        
        layout.addWidget(task_panel)
        
        # 创建分割布局
        main_split = QHBoxLayout()
        
        # 左侧：服务器状态表
        status_group = QGroupBox("服务器状态")
        status_layout = QVBoxLayout(status_group)
        
        # 创建表格显示服务器状态
        self.status_table = QTableWidget(5, 5)
        self.status_table.setHorizontalHeaderLabels(['ID', 'CPU使用率', '内存使用率', '任务数', '状态'])
        self.status_table.setShowGrid(True)  # 显示网格线
        self.status_table.setGridStyle(Qt.PenStyle.SolidLine)  # 实线网格
        self.status_table.setStyleSheet("QTableWidget { gridline-color: lightgray; }")  # 浅灰色网格线
        
        for i in range(5):
            self.update_server_row(i)
        
        status_layout.addWidget(self.status_table)
        main_split.addWidget(status_group)
        
        # 右侧：可视化图表
        viz_group = QGroupBox("可视化图表")
        viz_layout = QVBoxLayout(viz_group)
        
        # 创建图形场景和视图（使用支持缩放的视图）
        self.scene = QGraphicsScene()
        self.graphics_view = ZoomableGraphicsView(self.scene)
        
        viz_layout.addWidget(self.graphics_view)
        main_split.addWidget(viz_group)
        
        layout.addLayout(main_split)
        
        # 底部：任务队列显示
        queue_group = QGroupBox("任务队列")
        queue_layout = QVBoxLayout(queue_group)
        
        self.task_table = QTableWidget(0, 4)
        self.task_table.setHorizontalHeaderLabels(['任务ID', '类型', '优先级', '分配服务器'])
        self.task_table.setShowGrid(True)  # 显示网格线
        self.task_table.setGridStyle(Qt.PenStyle.SolidLine)  # 实线网格
        self.task_table.setStyleSheet("QTableWidget { gridline-color: lightgray; }")  # 浅灰色网格线
        queue_layout.addWidget(self.task_table)
        
        layout.addWidget(queue_group)
        
    def setup_timers(self):
        """设置定时器以更新可视化"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(500)  # 每0.5秒更新一次
        
    def add_task(self):
        """手动添加任务到后端"""
        # 获取任务参数
        task_type = self.task_type_combo.currentText()
        priority_text = self.priority_combo.currentText()
        cpu_demand = self.cpu_spinbox.value()
        memory_demand = self.memory_spinbox.value()
        
        # 将中文优先级转换为数值
        priority_map = {"高": 5, "中": 3, "低": 1}
        priority_value = priority_map[priority_text]
        
        # 创建任务对象
        task = {
            'cpu_req': float(cpu_demand),
            'memory_req': float(memory_demand),
            'priority': priority_value,
            'type': task_type
        }
        
        # 发送到后端控制器
        if hasattr(self, 'backend_controller') and self.backend_controller:
            self.backend_controller.add_task_to_env(task)
        else:
            viz_logger.debug(f"添加任务: 类型={task_type}, 优先级={priority_text}, CPU={cpu_demand}, 内存={memory_demand}")
        
        # 添加到本地任务列表用于显示
        task_id = f"TASK_{len(self.local_tasks) + 1:03d}"
        self.local_tasks.append({
            'id': task_id,
            'type': task_type,
            'priority': priority_text,
            'cpu_req': cpu_demand,
            'memory_req': memory_demand,
            'status': 'pending'
        })
        
        # 更新任务表
        self.task_table.setRowCount(len(self.local_tasks))
        row = len(self.local_tasks) - 1
        task = self.local_tasks[row]
        
        self.task_table.setItem(row, 0, QTableWidgetItem(task['id']))
        self.task_table.setItem(row, 1, QTableWidgetItem(task['type']))
        self.task_table.setItem(row, 2, QTableWidgetItem(task['priority']))
        self.task_table.setItem(row, 3, QTableWidgetItem("-"))  # 初始未分配
        
    def __init__(self):
        super().__init__()
        self.dummy_servers_data = [
            {'id': i, 'cpu': 0, 'memory': 0, 'tasks': 0, 'status': 'idle'}
            for i in range(5)
        ]
        self.servers_data = self.dummy_servers_data
        self.local_tasks = []  # 本地任务列表
        self.tasks_data = []  # 任务数据列表 - 修复缺失属性
        self.init_ui()
        self.setup_timers()
    
    def update_from_backend(self, data):
        """从后端更新数据"""
        if data:  # 如果有数据，更新显示
            self.servers_data = data
            self.update_tables()
        else:  # 如果数据为空，重置到初始状态
            # 重置服务器数据为初始状态
            self.servers_data = [
                {
                    'id': i,
                    'cpu': 0.0,
                    'memory': 0.0,
                    'status': 'idle',
                    'tasks_completed': 0,
                    'tasks_failed': 0,
                    'current_tasks': [],
                    'cpu_capacity': 100.0,
                    'memory_capacity': 100.0
                } for i in range(5)
            ]
            # 清空任务列表
            self.tasks_data = []
            # 清空本地任务列表（这是显示在任务表格中的）
            if hasattr(self, 'local_tasks'):
                self.local_tasks = []
            # 清空任务表格
            if hasattr(self, 'task_table') and self.task_table:
                self.task_table.setRowCount(0)
            self.update_tables()
    
    def update_tables(self):
        """更新表格显示"""
        for i in range(min(len(self.servers_data), 5)):
            self.update_server_row(i)
            
        # 更新任务队列 - 从服务器环境获取实际任务数据
        # 这里我们显示待处理任务和已分配任务
        all_tasks = []
        
        # 先处理待处理任务（未分配）
        pending_tasks = []
        if len(self.servers_data) > 0 and 'pending_tasks' in self.servers_data[0]:
            pending_tasks = self.servers_data[0].get('pending_tasks', [])
        
        # 添加待处理任务到列表
        for i, task in enumerate(pending_tasks):
            # 将数字优先级转换为中文显示
            priority_num = task.get('priority', 'Unknown')
            if isinstance(priority_num, (int, float)):
                if priority_num >= 4:
                    priority_display = "高"
                elif priority_num >= 2:
                    priority_display = "中"
                else:
                    priority_display = "低"
            else:
                priority_display = priority_num  # 如果已经是字符串，直接使用
            
            # 将内部任务类型转换为中文显示
            task_type = task.get('type', 'Unknown')
            if task_type == 'compute':
                type_display = "计算任务"
            elif task_type == 'storage':
                type_display = "存储任务"
            elif task_type == 'network':
                type_display = "网络任务"
            else:
                type_display = task_type  # 如果已经是中文或其他格式，直接使用
            
            all_tasks.append({
                'id': task.get('id', f'pending_{i}'),
                'type': type_display,
                'priority': priority_display,
                'assigned_server': '-'  # 未分配
            })
        
        # 处理已分配任务（在各个服务器上）
        for server_idx, server_info in enumerate(self.servers_data):
            current_tasks = server_info.get('current_tasks', [])
            for task in current_tasks:
                # 将数字优先级转换为中文显示
                priority_num = task.get('priority', 'Unknown')
                if isinstance(priority_num, (int, float)):
                    if priority_num >= 4:
                        priority_display = "高"
                    elif priority_num >= 2:
                        priority_display = "中"
                    else:
                        priority_display = "低"
                else:
                    priority_display = priority_num  # 如果已经是字符串，直接使用
                
                # 将内部任务类型转换为中文显示
                task_type = task.get('type', 'Unknown')
                if task_type == 'compute':
                    type_display = "计算任务"
                elif task_type == 'storage':
                    type_display = "存储任务"
                elif task_type == 'network':
                    type_display = "网络任务"
                else:
                    type_display = task_type  # 如果已经是中文或其他格式，直接使用
                
                all_tasks.append({
                    'id': task.get('id', f'server_{server_idx}_task'),
                    'type': type_display,
                    'priority': priority_display,
                    'assigned_server': f'Server {server_idx}'  # 已分配到服务器
                })
        
        # 同时也要处理本地任务列表，更新分配状态
        for local_task in self.local_tasks:
            # 检查该任务是否在待处理或已分配列表中
            task_found = False
            for task in all_tasks:
                if task['id'] == local_task['id']:
                    task_found = True
                    break
            
            # 如果任务不在all_tasks中，添加它（可能是新添加的）
            if not task_found:
                all_tasks.append({
                    'id': local_task['id'],
                    'type': local_task.get('type', 'Unknown'),
                    'priority': local_task.get('priority', 'Unknown'),
                    'assigned_server': local_task.get('assigned_server', '-')  # 初始未分配
                })
        
        # 设置任务表行数
        self.task_table.setRowCount(len(all_tasks))
        for i, task in enumerate(all_tasks):
            self.task_table.setItem(i, 0, QTableWidgetItem(str(task.get('id', ''))))
            self.task_table.setItem(i, 1, QTableWidgetItem(task.get('type', '')))
            self.task_table.setItem(i, 2, QTableWidgetItem(str(task.get('priority', ''))))
            self.task_table.setItem(i, 3, QTableWidgetItem(str(task.get('assigned_server', '-'))))
    
    def update_display(self, data):
        """更新显示（兼容方法名）"""
        self.update_from_backend(data)
        
    def update_server_row(self, row):
        """更新服务器状态表格中的行"""
        server = self.servers_data[row]
        
        # ID
        self.status_table.setItem(row, 0, QTableWidgetItem(f"Server-{server.get('id', row)}"))
        
        # CPU使用率
        cpu = server.get('cpu', 0)
        cpu_item = QTableWidgetItem(f"{cpu:.1f}%")
        cpu_progress = QProgressBar()
        cpu_progress.setValue(int(cpu))
        cpu_progress.setTextVisible(True)
        self.status_table.setCellWidget(row, 1, cpu_progress)
        
        # 内存使用率
        mem = server.get('memory', 0)
        mem_item = QTableWidgetItem(f"{mem:.1f}%")
        mem_progress = QProgressBar()
        mem_progress.setValue(int(mem))
        mem_progress.setTextVisible(True)
        self.status_table.setCellWidget(row, 2, mem_progress)
        
        # 任务数
        self.status_table.setItem(row, 3, QTableWidgetItem(str(server.get('tasks', 0))))
        
        # 状态
        status = server.get('status', 'idle')
        status_item = QTableWidgetItem(status)
        if status == 'busy':
            status_item.setBackground(QColor(255, 100, 100))  # 红色表示忙碌
        elif status == 'warning':
            status_item.setBackground(QColor(255, 255, 100))  # 黄色表示警告
        else:
            status_item.setBackground(QColor(100, 255, 100))  # 绿色表示空闲
        self.status_table.setItem(row, 4, status_item)
        
    def update_visualization(self):
        """更新可视化图表 - 仅更新图形显示，数据由后端提供"""
        # 更新表格（数据已由update_from_backend方法更新）
        for i in range(min(len(self.servers_data), 5)):
            self.update_server_row(i)
        
        # 更新图形场景
        self.draw_server_network()
    
    def draw_server_network(self):
        """绘制服务器网络图"""
        self.scene.clear()
        
        # 设置场景大小
        self.scene.setSceneRect(0, 0, 600, 400)
        
        # 服务器位置
        server_positions = [
            (100, 100), (300, 80), (500, 150),
            (200, 300), (450, 320)
        ]
        
        # 绘制服务器节点
        server_items = []
        for i, (x, y) in enumerate(server_positions):
            server = self.servers_data[i]
            
            # 根据状态设置颜色
            if server['status'] == 'busy':
                color = QColor(255, 100, 100)  # 红色
            elif server['status'] == 'warning':
                color = QColor(255, 255, 100)  # 黄色
            else:
                color = QColor(100, 255, 100)  # 绿色
            
            # 创建服务器圆圈
            circle = QGraphicsEllipseItem(x-30, y-30, 60, 60)
            circle.setBrush(QBrush(color))
            circle.setPen(QPen(QColor(0, 0, 0), 2))
            self.scene.addItem(circle)
            
            # 添加服务器标签
            text = QGraphicsTextItem(f"S{i}\nCPU:{server['cpu']:.0f}%\nMem:{server['memory']:.0f}%")
            text.setPos(x-25, y-10)
            text.setFont(QFont("Arial", 8))
            self.scene.addItem(text)
            
            server_items.append((circle, text, x, y))
        
        # 绘制连接线（模拟服务器间的通信）
        connections = [(0, 1), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4)]
        for i, j in connections:
            x1, y1 = server_positions[i]
            x2, y2 = server_positions[j]
            
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(QColor(200, 200, 200), 1))
            self.scene.addItem(line)


if __name__ == '__main__':
    # 测试服务器可视化组件
    from PyQt6.QtWidgets import QApplication
    from PyQt6 import QtGui
    
    app = QApplication(sys.argv)
    widget = ServerVisualizationWidget()
    widget.resize(1000, 700)
    widget.show()
    sys.exit(app.exec())