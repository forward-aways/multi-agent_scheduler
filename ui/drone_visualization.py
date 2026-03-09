"""
多智能体任务调度系统无人机任务可视化组件
实现无人机巡检和队形任务的可视化
"""

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QProgressBar, QTableWidget, QTableWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem, QComboBox, QPushButton,
    QDoubleSpinBox, QScrollArea, QDialog, QDialogButtonBox, QGridLayout,
    QTabWidget, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QBrush, QColor, QPen, QFont, QPainter
import random
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLScatterPlotItem, GLLinePlotItem, GLAxisItem, GLTextItem
from utils.logging_config import ProjectLogger

drone_mission_viz_logger = ProjectLogger('drone_mission_viz', log_dir='logs')


class ZoomableGraphicsView(QGraphicsView):
    """支持缩放的图形视图"""
    
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # 设置视口更新模式，避免拖动时产生残影
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        # 设置背景颜色
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        self.scale_factor = 1.15
        
    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale(self.scale_factor, self.scale_factor)
            else:
                self.scale(1 / self.scale_factor, 1 / self.scale_factor)
            event.accept()
        else:
            super().wheelEvent(event)


class DronePositionDialog(QDialog):
    """无人机位置设置对话框（支持任务类型选择）"""
    
    def __init__(self, drones_data, parent=None):
        super().__init__(parent)
        self.drones_data = drones_data
        self.drone_position_inputs = []
        self.waypoint_inputs = []
        self.setWindowTitle("无人机任务位置设置")
        self.setMinimumWidth(650)
        self.setMinimumHeight(500)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
                font-size: 12px;
            }
            QDoubleSpinBox {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 4px;
                font-size: 12px;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #0078d4;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 4px;
                font-size: 12px;
            }
            QComboBox:focus {
                border: 2px solid #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QDialogButtonBox QPushButton {
                min-width: 80px;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                padding: 8px 20px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0078d4;
                font-weight: bold;
            }
        """)
        self.init_ui()
    
    def init_ui(self):
        """初始化对话框界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 任务类型选择
        task_type_group = QGroupBox("任务类型")
        task_type_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #333333;
                border: 2px solid #0078d4;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        task_type_layout = QHBoxLayout(task_type_group)
        
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["队形任务", "协同包围任务"])  # 已移除巡检任务
        self.task_type_combo.currentTextChanged.connect(self.on_task_type_changed)
        task_type_layout.addWidget(QLabel("选择任务类型:"))
        task_type_layout.addWidget(self.task_type_combo)
        task_type_layout.addStretch()
        
        layout.addWidget(task_type_group)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
        """)
        
        # 队形任务选项卡
        self.formation_tab = QWidget()
        self.init_formation_tab()
        self.tab_widget.addTab(self.formation_tab, "队形任务")
        
        # 协同包围任务选项卡
        self.encirclement_tab = QWidget()
        self.init_encirclement_tab()
        self.tab_widget.addTab(self.encirclement_tab, "协同包围任务")
        
        layout.addWidget(self.tab_widget)
        
        # 添加提示信息
        hint_label = QLabel("💡 提示: 双击数字可直接编辑，或使用键盘上下键调整数值")
        hint_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                padding: 5px;
                background-color: #fff9e6;
                border-radius: 3px;
                border: 1px solid #ffe066;
            }
        """)
        layout.addWidget(hint_label)
        
        # 添加按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet("""
            QDialogButtonBox {
                background-color: transparent;
            }
        """)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def init_inspection_tab(self):
        """初始化巡检任务选项卡"""
        layout = QGridLayout(self.inspection_tab)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 表头样式
        header_style = """
            QLabel {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
                border-radius: 3px;
            }
        """
        
        # 添加表头
        for col, label in enumerate(["位置", "X坐标", "Y坐标", "Z坐标"]):
            header = QLabel(label)
            header.setStyleSheet(header_style)
            layout.addWidget(header, 0, col)
        
        # 起点
        layout.addWidget(QLabel("起点"), 1, 0)
        self.start_x = self.create_spinbox(10, 0, 100)
        self.start_y = self.create_spinbox(10, 0, 100)
        self.start_z = self.create_spinbox(10, 0, 50)
        layout.addWidget(self.start_x, 1, 1)
        layout.addWidget(self.start_y, 1, 2)
        layout.addWidget(self.start_z, 1, 3)
        
        # 终点
        layout.addWidget(QLabel("终点"), 2, 0)
        self.end_x = self.create_spinbox(90, 0, 100)
        self.end_y = self.create_spinbox(90, 0, 100)
        self.end_z = self.create_spinbox(40, 0, 50)
        layout.addWidget(self.end_x, 2, 1)
        layout.addWidget(self.end_y, 2, 2)
        layout.addWidget(self.end_z, 2, 3)
        
        # 检查点
        waypoint_labels = ["检查点A", "检查点B", "检查点C", "检查点D"]
        waypoint_defaults = [
            [30, 30, 20],
            [50, 50, 25],
            [70, 30, 20],
            [50, 70, 30]
        ]
        
        for i, (label, defaults) in enumerate(zip(waypoint_labels, waypoint_defaults)):
            row = i + 3
            layout.addWidget(QLabel(label), row, 0)
            x = self.create_spinbox(defaults[0], 0, 100)
            y = self.create_spinbox(defaults[1], 0, 100)
            z = self.create_spinbox(defaults[2], 0, 50)
            layout.addWidget(x, row, 1)
            layout.addWidget(y, row, 2)
            layout.addWidget(z, row, 3)
            self.waypoint_inputs.append([x, y, z])
        
        layout.setRowStretch(7, 1)
    
    def init_formation_tab(self):
        """初始化队形任务选项卡"""
        layout = QGridLayout(self.formation_tab)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 表头样式
        header_style = """
            QLabel {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
                border-radius: 3px;
            }
        """
        
        # 添加表头
        for col, label in enumerate(["位置", "X坐标", "Y坐标", "Z坐标"]):
            header = QLabel(label)
            header.setStyleSheet(header_style)
            layout.addWidget(header, 0, col)
        
        # 起点
        layout.addWidget(QLabel("起点"), 1, 0)
        self.formation_start_x = self.create_spinbox(20, 0, 100)
        self.formation_start_y = self.create_spinbox(20, 0, 100)
        self.formation_start_z = self.create_spinbox(15, 0, 50)
        layout.addWidget(self.formation_start_x, 1, 1)
        layout.addWidget(self.formation_start_y, 1, 2)
        layout.addWidget(self.formation_start_z, 1, 3)
        
        # 终点
        layout.addWidget(QLabel("终点"), 2, 0)
        self.formation_end_x = self.create_spinbox(80, 0, 100)
        self.formation_end_y = self.create_spinbox(80, 0, 100)
        self.formation_end_z = self.create_spinbox(35, 0, 50)
        layout.addWidget(self.formation_end_x, 2, 1)
        layout.addWidget(self.formation_end_y, 2, 2)
        layout.addWidget(self.formation_end_z, 2, 3)
        
        # 添加说明
        hint = QLabel("队形任务：无人机从起点出发，自主规划路径到达终点，同时保持队形")
        hint.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                padding: 10px;
                background-color: #e6f7ff;
                border-radius: 5px;
                border: 1px solid #91d5ff;
            }
        """)
        hint.setWordWrap(True)
        layout.addWidget(hint, 3, 0, 1, 4)
        
        layout.setRowStretch(4, 1)
    
    def init_encirclement_tab(self):
        """初始化协同包围任务选项卡"""
        layout = QGridLayout(self.encirclement_tab)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 表头样式
        header_style = """
            QLabel {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 8px;
                border-radius: 3px;
            }
        """
        
        # 添加表头
        for col, label in enumerate(["位置", "X坐标", "Y坐标", "Z坐标"]):
            header = QLabel(label)
            header.setStyleSheet(header_style)
            layout.addWidget(header, 0, col)
        
        # 目标位置
        layout.addWidget(QLabel("目标位置"), 1, 0)
        self.encirclement_target_x = self.create_spinbox(50, 0, 100)
        self.encirclement_target_y = self.create_spinbox(50, 0, 100)
        self.encirclement_target_z = self.create_spinbox(25, 0, 50)
        layout.addWidget(self.encirclement_target_x, 1, 1)
        layout.addWidget(self.encirclement_target_y, 1, 2)
        layout.addWidget(self.encirclement_target_z, 1, 3)
        
        # 目标速度
        layout.addWidget(QLabel("目标速度"), 2, 0)
        self.encirclement_target_vx = self.create_spinbox(0.3, -1.0, 1.0)
        self.encirclement_target_vy = self.create_spinbox(0.3, -1.0, 1.0)
        self.encirclement_target_vz = self.create_spinbox(0.0, -1.0, 1.0)
        layout.addWidget(self.encirclement_target_vx, 2, 1)
        layout.addWidget(self.encirclement_target_vy, 2, 2)
        layout.addWidget(self.encirclement_target_vz, 2, 3)
        
        # 添加说明
        hint = QLabel("协同包围任务：无人机从不同方向包围移动目标，保持20米包围半径，持续50步以上")
        hint.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 11px;
                padding: 10px;
                background-color: #e6f7ff;
                border-radius: 5px;
                border: 1px solid #91d5ff;
            }
        """)
        hint.setWordWrap(True)
        layout.addWidget(hint, 3, 0, 1, 4)
        
        layout.setRowStretch(4, 1)
    
    def create_spinbox(self, value, min_val, max_val):
        """创建数字输入框"""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        spinbox.setFixedWidth(100)
        return spinbox
    
    def on_task_type_changed(self, text):
        """任务类型改变时的处理"""
        if text == "队形任务":
            self.tab_widget.setCurrentIndex(0)
        elif text == "协同包围任务":
            self.tab_widget.setCurrentIndex(1)
    
    def get_positions(self):
        """获取用户设置的位置信息"""
        task_type = self.task_type_combo.currentText()
        
        if task_type == "队形任务":
            return {
                'task_type': 'formation',
                'start_point': [self.formation_start_x.value(), self.formation_start_y.value(), self.formation_start_z.value()],
                'end_point': [self.formation_end_x.value(), self.formation_end_y.value(), self.formation_end_z.value()]
            }
        elif task_type == "协同包围任务":
            return {
                'task_type': 'encirclement',
                'target_position': [self.encirclement_target_x.value(), self.encirclement_target_y.value(), self.encirclement_target_z.value()],
                'target_velocity': [self.encirclement_target_vx.value(), self.encirclement_target_vy.value(), self.encirclement_target_vz.value()]
            }
        
        return None


class DroneMissionVisualizationWidget(QWidget):
    """无人机任务可视化组件（巡检和队形）"""
    
    def __init__(self):
        super().__init__()
        # 设置默认起点和终点（与训练代码一致）
        self.start_point = [15.0, 15.0, 15.0]
        self.end_point = [85.0, 85.0, 35.0]
        
        # 设置默认检查点（与训练代码一致）
        self.waypoints = [
            [30.0, 30.0, 20.0],   # 检查点1
            [50.0, 50.0, 25.0],   # 检查点2
            [70.0, 70.0, 30.0],   # 检查点3
            [40.0, 60.0, 22.0]    # 检查点4
        ]
        
        # 三个无人机的初始位置（在起点附近）
        self.drones_data = [
            {
                'id': 0,
                'position': [12.0, 12.0, 12.0],  # 无人机0：起点附近
                'battery': 100.0,
                'speed': 0.0,
                'task_progress': 0.0,
                'status': 'idle'
            },
            {
                'id': 1,
                'position': [15.0, 12.0, 12.0],  # 无人机1：起点附近
                'battery': 100.0,
                'speed': 0.0,
                'task_progress': 0.0,
                'status': 'idle'
            },
            {
                'id': 2,
                'position': [13.5, 15.0, 12.0],  # 无人机2：起点附近
                'battery': 100.0,
                'speed': 0.0,
                'task_progress': 0.0,
                'status': 'idle'
            }
        ]
        
        self.task_type = 'formation'  # 默认为队形任务
        self.inspection_path = []
        self.formation_targets = []
        self.target_position = [50, 50, 25]
        self.target_velocity = [0.3, 0.3, 0.0]
        self.encirclement_radius = 20.0
        self.encirclement_time = 0
        self.encirclement_success = False
        self.drone_scatter_items = []
        self.waypoint_items = []
        self.start_point_items = []
        self.end_point_items = []
        self.formation_target_items = []
        self.path_line_items = []
        self.drone_text_items = []
        self.formation_text_items = []
        self.leader_drone_idx = 0
        self.formation_start = None
        self.formation_end = None
        self.formation_path = []
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        title = QLabel("无人机任务调度系统")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 创建控制面板
        control_group = QGroupBox("任务控制")
        control_layout = QHBoxLayout(control_group)
        
        control_layout.addWidget(QLabel("任务类型:"))
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["队形任务", "协同包围任务"])  # 已移除巡检任务
        self.task_type_combo.currentTextChanged.connect(self.change_task_type)
        control_layout.addWidget(self.task_type_combo)
        
        control_layout.addWidget(QLabel("队形类型:"))
        self.formation_combo = QComboBox()
        self.formation_combo.addItems(["三角形", "V形", "一字形"])
        self.formation_combo.currentTextChanged.connect(self.change_formation_type)
        control_layout.addWidget(self.formation_combo)
        
        start_btn = QPushButton("开始任务")
        start_btn.clicked.connect(self.start_mission)
        control_layout.addWidget(start_btn)
        
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_mission)
        control_layout.addWidget(reset_btn)
        
        position_btn = QPushButton("设置位置")
        position_btn.clicked.connect(self.open_position_dialog)
        control_layout.addWidget(position_btn)
        
        control_layout.addStretch()
        layout.addWidget(control_group)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 2D视图标签页
        self.create_2d_tab(tab_widget)
        
        # 3D视图标签页
        self.create_3d_tab(tab_widget)
        
        layout.addWidget(tab_widget)
        
        # 底部：任务信息
        info_group = QGroupBox("任务信息")
        info_layout = QVBoxLayout(info_group)
        
        self.task_info_label = QLabel("当前任务: 无")
        self.task_info_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.task_info_label)
        
        layout.addWidget(info_group)
        
    def create_2d_tab(self, tab_widget):
        """创建2D视图标签页"""
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：无人机状态表
        status_group = QGroupBox("无人机状态")
        status_layout = QVBoxLayout(status_group)
        
        self.status_table = QTableWidget(3, 6)
        self.status_table.setHorizontalHeaderLabels(['ID', '位置(X,Y,Z)', '电量', '速度', '任务进度', '状态'])
        
        for i in range(3):
            self.update_drone_row(i)
        
        status_layout.addWidget(self.status_table)
        
        # 右侧：任务地图
        map_group = QGroupBox("任务地图 (2D)")
        map_layout = QVBoxLayout(map_group)
        
        self.scene = QGraphicsScene()
        self.graphics_view = ZoomableGraphicsView(self.scene)
        
        map_layout.addWidget(self.graphics_view)
        
        # 添加组件到分割器
        splitter.addWidget(status_group)
        splitter.addWidget(map_group)
        
        # 设置初始尺寸比例
        splitter.setSizes([250, 750])
        
        # 创建2D容器
        container_2d = QWidget()
        container_layout = QVBoxLayout(container_2d)
        container_layout.addWidget(splitter)
        
        tab_widget.addTab(container_2d, "2D视图")
    
    def create_3d_tab(self, tab_widget):
        """创建3D视图标签页"""
        # 创建分割器
        splitter_3d = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：无人机状态表
        status_group = QGroupBox("无人机状态")
        status_group.setMinimumWidth(180)
        status_layout = QVBoxLayout(status_group)
        
        self.status_table_3d = QTableWidget(3, 6)
        self.status_table_3d.setHorizontalHeaderLabels(['ID', '位置', '电量', '速度', '进度', '状态'])
        self.status_table_3d.setMinimumWidth(180)
        
        for i in range(3):
            self.update_drone_row_3d(i)
        
        status_layout.addWidget(self.status_table_3d)
        
        # 右侧：3D任务地图
        map_group = QGroupBox("任务地图 (3D)")
        map_layout = QVBoxLayout(map_group)
        
        # 创建3D视图
        self.view_3d = GLViewWidget()
        self.view_3d.setSizePolicy(self.sizePolicy())
        
        # 添加坐标轴
        axis = GLAxisItem()
        axis.setSize(x=100, y=100, z=50)
        self.view_3d.addItem(axis)
        
        # 添加网格
        grid = GLGridItem()
        grid.setSize(x=100, y=100)
        grid.setSpacing(x=5, y=5)
        self.view_3d.addItem(grid)
        
        # 设置相机视角（调整距离让视图更大）
        self.view_3d.setCameraPosition(distance=250, elevation=30, azimuth=45)
        
        map_layout.addWidget(self.view_3d)
        
        # 添加组件到分割器
        splitter_3d.addWidget(status_group)
        splitter_3d.addWidget(map_group)
        
        # 设置初始尺寸比例
        splitter_3d.setSizes([250, 1000])
        
        # 创建3D容器
        container_3d = QWidget()
        container_layout = QVBoxLayout(container_3d)
        container_layout.addWidget(splitter_3d)
        
        tab_widget.addTab(container_3d, "3D视图")
    
    def setup_timers(self):
        """设置定时器"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(500)
    
    def change_task_type(self, task_type_text):
        """切换任务类型"""
        if task_type_text == "协同包围任务":
            self.task_type = 'encirclement'
            self.formation_combo.setEnabled(False)
            # 设置默认的目标位置和速度
            self.target_position = [50, 50, 25]
            self.target_velocity = [0.3, 0.3, 0.0]
            self.encirclement_radius = 20.0
            self.encirclement_time = 0
            self.encirclement_success = False
        else:
            self.task_type = 'formation'
            self.formation_combo.setEnabled(True)
        
        drone_mission_viz_logger.info(f"任务类型切换为: {task_type_text}")
        
        if hasattr(self, 'backend_controller') and self.backend_controller:
            self.backend_controller.change_drone_task_type(self.task_type)
        
        # 更新可视化
        self.update_visualization()
    
    def change_formation_type(self, formation_text):
        """切换队形类型"""
        formation_map = {
            "三角形": "triangle",
            "V形": "v_shape",
            "一字形": "line"
        }
        formation_type = formation_map[formation_text]
        
        drone_mission_viz_logger.info(f"队形类型切换为: {formation_text}")
        
        if hasattr(self, 'backend_controller') and self.backend_controller:
            self.backend_controller.change_drone_formation_type(formation_type)
    
    def start_mission(self):
        """开始任务"""
        drone_mission_viz_logger.info("开始任务")
        
        if hasattr(self, 'backend_controller') and self.backend_controller:
            self.backend_controller.start_drone_mission()
    
    def reset_mission(self):
        """重置任务"""
        drone_mission_viz_logger.info("重置任务")
        
        if hasattr(self, 'backend_controller') and self.backend_controller:
            self.backend_controller.reset_drone_mission()
    
    def open_position_dialog(self):
        """打开无人机位置设置对话框"""
        dialog = DronePositionDialog(self.drones_data, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            positions = dialog.get_positions()
            self.apply_drone_positions(positions)
    
    def apply_drone_positions(self, positions):
        """应用用户设置的无人机位置"""
        # 更新可视化中的位置信息
        if positions['task_type'] == 'inspection':
            self.start_point = positions['start_point']
            self.end_point = positions['end_point']
            self.waypoints = [np.array(wp) for wp in positions['waypoints']]
            drone_mission_viz_logger.info(f"巡检任务位置已设置: 起点={self.start_point}, 终点={self.end_point}")
        elif positions['task_type'] == 'formation':
            self.formation_start = positions['start_point']
            self.formation_end = positions['end_point']
            drone_mission_viz_logger.info(f"队形任务位置已设置: 起点={self.formation_start}, 终点={self.formation_end}")
        
        # 更新可视化
        self.update_visualization()
        
        # 发送到后端控制器
        if hasattr(self, 'backend_controller') and self.backend_controller is not None:
            self.backend_controller.set_custom_drone_positions(positions)
            drone_mission_viz_logger.info("位置信息已发送到后端控制器")
    
    def update_from_backend(self, data):
        """从后端控制器更新数据"""
        if data and isinstance(data, dict):
            # 新的数据结构：包含任务信息
            task_type = data.get('task_type', 'inspection')
            drones = data.get('drones', [])
            
            if drones and len(drones) > 0:
                self.drones_data = drones
                
                # 巡检任务信息
                if task_type == 'inspection':
                    # 如果后端发送了起点、终点和检查点，使用后端数据
                    if 'start_point' in data:
                        self.start_point = data['start_point']
                    if 'end_point' in data:
                        self.end_point = data['end_point']
                    if 'waypoints' in data:
                        self.waypoints = data['waypoints']
                    if 'inspection_path' in data:
                        self.inspection_path = data['inspection_path']
                    else:
                        # 构建完整路径
                        self.inspection_path = [self.start_point] + self.waypoints + [self.end_point] + list(reversed(self.waypoints)) + [self.start_point]
                
                # 队形任务信息
                elif task_type == 'formation':
                    self.leader_drone_idx = data.get('leader_drone_idx', 0)
                    self.formation_start = data.get('formation_start', None)
                    self.formation_end = data.get('formation_end', None)
                    formation_offsets = data.get('formation_offsets', [])
                    if formation_offsets:
                        # 根据领航机位置和偏移量计算队形目标位置
                        leader_pos = self.drones_data[self.leader_drone_idx].get('position', [0, 0, 0])
                        self.formation_targets = [
                            [leader_pos[0] + offset[0], leader_pos[1] + offset[1], leader_pos[2] + offset[2]]
                            for offset in formation_offsets
                        ]
                
                # 协同包围任务信息
                elif task_type == 'encirclement':
                    self.target_position = data.get('target_position', [50, 50, 25])
                    self.target_velocity = data.get('target_velocity', [0.3, 0.3, 0.0])
                    self.encirclement_radius = data.get('encirclement_radius', 20.0)
                    self.encirclement_time = data.get('encirclement_time', 0)
                    self.encirclement_success = data.get('encirclement_success', False)
                
                self.update_tables()
            else:
                self.drones_data = [
                    {
                        'id': i,
                        'position': [0.0, 0.0, 0.0],
                        'battery': 100.0,
                        'speed': 0.0,
                        'task_progress': 0.0,
                        'status': 'idle'
                    } for i in range(3)
                ]
                self.waypoints = []
                self.start_point = None
                self.end_point = None
                self.inspection_path = []
                self.formation_targets = []
                self.target_position = [50, 50, 25]
                self.target_velocity = [0.3, 0.3, 0.0]
                self.encirclement_radius = 20.0
                self.encirclement_time = 0
                self.encirclement_success = False
                self.update_tables()
        elif data and isinstance(data, list):
            # 旧的数据结构（向后兼容）
            self.drones_data = data
            self.update_tables()
        else:
            self.drones_data = [
                {
                    'id': i,
                    'position': [0.0, 0.0, 0.0],
                    'battery': 100.0,
                    'speed': 0.0,
                    'task_progress': 0.0,
                    'status': 'idle'
                } for i in range(3)
            ]
            # 使用默认的起点、终点和检查点
            # 构建完整路径：起点 → A → B → C → 终点 → C → B → A → 起点
            self.inspection_path = [self.start_point] + self.waypoints + [self.end_point] + list(reversed(self.waypoints)) + [self.start_point]
            self.formation_targets = []
            self.target_position = [50, 50, 25]
            self.target_velocity = [0.3, 0.3, 0.0]
            self.encirclement_radius = 20.0
            self.encirclement_time = 0
            self.encirclement_success = False
            self.update_tables()
    
    def update_display(self, data):
        """更新显示（兼容方法名）"""
        self.update_from_backend(data)
    
    def update_tables(self):
        """更新表格显示"""
        for i in range(min(len(self.drones_data), 3)):
            self.update_drone_row(i)
            self.update_drone_row_3d(i)
    
    def update_drone_row(self, row):
        """更新2D视图中的无人机状态表格"""
        if row >= len(self.drones_data):
            return
            
        drone = self.drones_data[row]
        
        self.status_table.setItem(row, 0, QTableWidgetItem(f"Drone-{drone.get('id', row)}"))
        
        position = drone.get('position', [0, 0, 0])
        pos_text = f"[{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}]"
        self.status_table.setItem(row, 1, QTableWidgetItem(pos_text))
        
        battery = drone.get('battery', 100)
        battery_progress = QProgressBar()
        battery_progress.setValue(int(battery))
        battery_progress.setTextVisible(True)
        self.status_table.setCellWidget(row, 2, battery_progress)
        
        speed = drone.get('speed', 0)
        self.status_table.setItem(row, 3, QTableWidgetItem(f"{speed:.1f} m/s"))
        
        progress = drone.get('task_progress', 0.0)
        self.status_table.setItem(row, 4, QTableWidgetItem(f"{progress*100:.1f}%"))
        
        status = drone.get('status', 'idle')
        status_item = QTableWidgetItem(status)
        if status == 'flying':
            status_item.setBackground(QColor(100, 150, 255))
        elif status == 'returning':
            status_item.setBackground(QColor(255, 200, 100))
        else:
            status_item.setBackground(QColor(100, 255, 100))
        self.status_table.setItem(row, 5, status_item)
    
    def update_drone_row_3d(self, row):
        """更新3D视图中的无人机状态表格"""
        if row >= len(self.drones_data):
            return
            
        drone = self.drones_data[row]
        
        self.status_table_3d.setItem(row, 0, QTableWidgetItem(f"Drone-{drone.get('id', row)}"))
        
        position = drone.get('position', [0, 0, 0])
        pos_text = f"[{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}]"
        self.status_table_3d.setItem(row, 1, QTableWidgetItem(pos_text))
        
        battery = drone.get('battery', 100)
        battery_progress = QProgressBar()
        battery_progress.setValue(int(battery))
        battery_progress.setTextVisible(True)
        self.status_table_3d.setCellWidget(row, 2, battery_progress)
        
        speed = drone.get('speed', 0)
        self.status_table_3d.setItem(row, 3, QTableWidgetItem(f"{speed:.1f} m/s"))
        
        progress = drone.get('task_progress', 0.0)
        self.status_table_3d.setItem(row, 4, QTableWidgetItem(f"{progress*100:.1f}%"))
        
        status = drone.get('status', 'idle')
        status_item = QTableWidgetItem(status)
        if status == 'flying':
            status_item.setBackground(QColor(100, 150, 255))
        elif status == 'returning':
            status_item.setBackground(QColor(255, 200, 100))
        else:
            status_item.setBackground(QColor(100, 255, 100))
        self.status_table_3d.setItem(row, 5, status_item)
    
    def update_visualization(self):
        """更新可视化图表"""
        # 更新 2D 视图
        self.scene.clear()
        
        if self.task_type == 'encirclement':
            self._draw_encirclement_view()
        else:  # formation
            self._draw_formation_view()
        
        # 更新 3D 视图
        self.update_3d_visualization()
    
    def update_3d_visualization(self):
        """更新3D可视化"""
        # 清除现有的3D项目
        for item in self.drone_scatter_items:
            self.view_3d.removeItem(item)
        self.drone_scatter_items = []
        
        for item in self.waypoint_items:
            self.view_3d.removeItem(item)
        self.waypoint_items = []
        
        for item in self.start_point_items:
            self.view_3d.removeItem(item)
        self.start_point_items = []
        
        for item in self.end_point_items:
            self.view_3d.removeItem(item)
        self.end_point_items = []
        
        for item in self.formation_target_items:
            self.view_3d.removeItem(item)
        self.formation_target_items = []
        
        for item in self.path_line_items:
            self.view_3d.removeItem(item)
        self.path_line_items = []
        
        # 清除文本标签
        for item in self.drone_text_items:
            self.view_3d.removeItem(item)
        self.drone_text_items = []
        
        for item in self.formation_text_items:
            self.view_3d.removeItem(item)
        self.formation_text_items = []
        
        # 绘制根据任务类型
        if self.task_type == 'encirclement':
            self._draw_3d_encirclement_view()
        else:  # formation
            self._draw_3d_formation_view()
    
    def _draw_inspection_view(self):
        """绘制巡检任务视图"""
        # 绘制起点（蓝色）
        if self.start_point is not None:
            sx, sy, sz = self.start_point
            screen_sx = sx * 5 + 100
            screen_sy = sy * 5 + 100
            
            start_ellipse = self.scene.addEllipse(screen_sx-8, screen_sy-8, 16, 16,
                                                 QPen(QColor(0, 100, 255), 3), QBrush(QColor(0, 100, 255, 100)))
            start_text = self.scene.addText("起点")
            start_text.setPos(screen_sx-15, screen_sy-25)
            start_text.setDefaultTextColor(QColor(0, 0, 255))
        
        # 绘制终点（红色）
        if self.end_point is not None:
            ex, ey, ez = self.end_point
            screen_ex = ex * 5 + 100
            screen_ey = ey * 5 + 100
            
            end_ellipse = self.scene.addEllipse(screen_ex-8, screen_ey-8, 16, 16,
                                               QPen(QColor(255, 0, 0), 3), QBrush(QColor(255, 0, 0, 100)))
            end_text = self.scene.addText("终点")
            end_text.setPos(screen_ex-15, screen_ey-25)
            end_text.setDefaultTextColor(QColor(200, 0, 0))
        
        # 绘制检查点（橙色空心圆圈）
        for i, waypoint in enumerate(self.waypoints):
            x, y, z = waypoint
            screen_x = x * 5 + 100
            screen_y = y * 5 + 100
            
            # 橙色
            color = QColor(255, 165, 0)
            # 空心圆圈：使用橙色边框，透明填充
            ellipse = self.scene.addEllipse(screen_x-8, screen_y-8, 16, 16,
                                           QPen(color, 3), QBrush(QColor(255, 165, 0, 0)))
            
            text = self.scene.addText(f"检查点{i+1}")
            text.setPos(screen_x-15, screen_y+10)
            text.setDefaultTextColor(QColor(0, 0, 0))
        
        # 绘制完整巡检路径（起点→检查点→终点→检查点→起点）
        if self.start_point is not None and self.end_point is not None and len(self.waypoints) > 0:
            # 构建完整路径
            full_path = [self.start_point] + self.waypoints + [self.end_point] + list(reversed(self.waypoints)) + [self.start_point]
            
            # 绘制路径线
            for i in range(len(full_path) - 1):
                x1, y1, z1 = full_path[i]
                x2, y2, z2 = full_path[i + 1]
                screen_x1 = x1 * 5 + 100
                screen_y1 = y1 * 5 + 100
                screen_x2 = x2 * 5 + 100
                screen_y2 = y2 * 5 + 100
                
                line = self.scene.addLine(screen_x1, screen_y1, screen_x2, screen_y2,
                                         QPen(QColor(0, 100, 255), 2))
        
        # 绘制无人机
        for i, drone in enumerate(self.drones_data):
            x, y, z = drone.get('position', [0, 0, 0])
            screen_x = x * 5 + 100
            screen_y = y * 5 + 100
            
            status = drone.get('status', 'idle')
            color = QColor(0, 255, 0)
            if status == 'flying':
                color = QColor(0, 100, 255)
            elif status == 'returning':
                color = QColor(255, 165, 0)
            
            ellipse = self.scene.addEllipse(screen_x-12, screen_y-12, 24, 24,
                                           QPen(color), QBrush(color))
            
            battery = drone.get('battery', 100)
            text = self.scene.addText(f"无人机{i}\n{battery:.0f}%")
            text.setPos(screen_x-15, screen_y+15)
            text.setDefaultTextColor(QColor(0, 0, 0))
    
    def _draw_encirclement_view(self):
        """绘制协同包围任务视图"""
        # 绘制包围半径（蓝色虚线圆圈）
        if hasattr(self, 'target_position') and self.target_position is not None:
            tx, ty, tz = self.target_position
            screen_tx = tx * 5 + 100
            screen_ty = ty * 5 + 100
            
            # 绘制包围半径圆圈
            radius_screen = self.encirclement_radius * 5
            circle = self.scene.addEllipse(screen_tx-radius_screen, screen_ty-radius_screen, 
                                          radius_screen*2, radius_screen*2,
                                          QPen(QColor(0, 100, 255), 2, Qt.PenStyle.DashLine),
                                          QBrush(QColor(0, 100, 255, 50)))
            
            # 绘制目标位置（红色）
            target_ellipse = self.scene.addEllipse(screen_tx-10, screen_ty-10, 20, 20,
                                                  QPen(QColor(255, 0, 0), 3), QBrush(QColor(255, 0, 0, 150)))
            target_text = self.scene.addText(f"目标\n{self.encirclement_time}步")
            target_text.setPos(screen_tx-20, screen_ty-35)
            target_text.setDefaultTextColor(QColor(200, 0, 0))
            
            # 绘制包围成功提示
            if hasattr(self, 'encirclement_success') and self.encirclement_success:
                success_text = self.scene.addText("包围成功！")
                success_text.setPos(screen_tx-40, screen_ty-60)
                success_text.setDefaultTextColor(QColor(0, 200, 0))
                success_text.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        # 绘制无人机
        for i, drone in enumerate(self.drones_data):
            x, y, z = drone.get('position', [0, 0, 0])
            screen_x = x * 5 + 100
            screen_y = y * 5 + 100
            
            status = drone.get('status', 'idle')
            color = QColor(0, 255, 0)
            if status == 'flying':
                color = QColor(0, 100, 255)
            
            ellipse = self.scene.addEllipse(screen_x-12, screen_y-12, 24, 24,
                                           QPen(color), QBrush(color))
            
            battery = drone.get('battery', 100)
            text = self.scene.addText(f"无人机{i}\n{battery:.0f}%")
            text.setPos(screen_x-15, screen_y+15)
            text.setDefaultTextColor(QColor(0, 0, 0))
    
    def _draw_formation_view(self):
        """绘制队形任务视图"""
        # 绘制起点（蓝色）
        if self.formation_start is not None:
            sx, sy, sz = self.formation_start
            screen_sx = sx * 5 + 100
            screen_sy = sy * 5 + 100
            
            start_ellipse = self.scene.addEllipse(screen_sx-8, screen_sy-8, 16, 16,
                                                 QPen(QColor(0, 100, 255), 3), QBrush(QColor(0, 100, 255, 100)))
            start_text = self.scene.addText("起点")
            start_text.setPos(screen_sx-15, screen_sy-25)
            start_text.setDefaultTextColor(QColor(0, 0, 255))
        
        # 绘制终点（红色）
        if self.formation_end is not None:
            ex, ey, ez = self.formation_end
            screen_ex = ex * 5 + 100
            screen_ey = ey * 5 + 100
            
            end_ellipse = self.scene.addEllipse(screen_ex-8, screen_ey-8, 16, 16,
                                               QPen(QColor(255, 0, 0), 3), QBrush(QColor(255, 0, 0, 100)))
            end_text = self.scene.addText("终点")
            end_text.setPos(screen_ex-15, screen_ey-25)
            end_text.setDefaultTextColor(QColor(200, 0, 0))
        
        # 绘制队形目标位置（橙色空心圆圈）
        for i, target in enumerate(self.formation_targets):
            x, y, z = target
            screen_x = x * 5 + 100
            screen_y = y * 5 + 100
            
            # 领航机用金色，僚机用红色
            if i == self.leader_drone_idx:
                color = QColor(255, 215, 0)
            else:
                color = QColor(255, 0, 0)
            
            # 空心圆圈
            ellipse = self.scene.addEllipse(screen_x-10, screen_y-10, 20, 20,
                                           QPen(color, 3), QBrush(Qt.BrushStyle.NoBrush))
            
            text = self.scene.addText(f"目标{i}")
            text.setPos(screen_x-15, screen_y+10)
            text.setDefaultTextColor(QColor(0, 0, 0))
        
        # 绘制队形连线
        if len(self.formation_targets) > 1:
            for i in range(len(self.formation_targets) - 1):
                x1, y1, z1 = self.formation_targets[i]
                x2, y2, z2 = self.formation_targets[i + 1]
                screen_x1 = x1 * 5 + 100
                screen_y1 = y1 * 5 + 100
                screen_x2 = x2 * 5 + 100
                screen_y2 = y2 * 5 + 100
                
                line = self.scene.addLine(screen_x1, screen_y1, screen_x2, screen_y2,
                                         QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine))
        
        # 绘制无人机
        for i, drone in enumerate(self.drones_data):
            x, y, z = drone.get('position', [0, 0, 0])
            screen_x = x * 5 + 100
            screen_y = y * 5 + 100
            
            status = drone.get('status', 'idle')
            color = QColor(0, 255, 0)
            if status == 'flying':
                color = QColor(0, 100, 255)
            elif status == 'returning':
                color = QColor(255, 165, 0)
            
            ellipse = self.scene.addEllipse(screen_x-12, screen_y-12, 24, 24,
                                           QPen(color), QBrush(color))
            
            battery = drone.get('battery', 100)
            text = self.scene.addText(f"无人机{i}\n{battery:.0f}%")
            text.setPos(screen_x-15, screen_y+15)
            text.setDefaultTextColor(QColor(0, 0, 0))
            
            # 绘制到目标位置的连线
            if i < len(self.formation_targets):
                target = self.formation_targets[i]
                tx, ty, tz = target
                screen_tx = tx * 5 + 100
                screen_ty = ty * 5 + 100
                
                line = self.scene.addLine(screen_x, screen_y, screen_tx, screen_ty,
                                         QPen(QColor(128, 128, 128), 1, Qt.PenStyle.DotLine))
    
    def _draw_3d_inspection_view(self):
        """绘制3D巡检任务视图（支持往返路径）"""
        # 绘制起点（蓝色）
        if self.start_point is not None:
            x, y, z = self.start_point
            start_scatter = GLScatterPlotItem(
                pos=np.array([[x, y, z]]),
                color=np.array([[0, 0, 1, 1]]),  # 蓝色
                size=[60],
                pxMode=False
            )
            self.view_3d.addItem(start_scatter)
            self.start_point_items.append(start_scatter)
            
            # 添加起点标签
            start_text = GLTextItem(pos=(x, y, z + 8), text="起点", color='white')
            self.view_3d.addItem(start_text)
            self.start_point_items.append(start_text)
        
        # 绘制终点（红色）
        if self.end_point is not None:
            x, y, z = self.end_point
            end_scatter = GLScatterPlotItem(
                pos=np.array([[x, y, z]]),
                color=np.array([[1, 0, 0, 1]]),  # 红色
                size=[60],
                pxMode=False
            )
            self.view_3d.addItem(end_scatter)
            self.end_point_items.append(end_scatter)
            
            # 添加终点标签
            end_text = GLTextItem(pos=(x, y, z + 8), text="终点", color='white')
            self.view_3d.addItem(end_text)
            self.end_point_items.append(end_text)
        
        # 绘制检查点（橙色空心圆圈）
        if len(self.waypoints) > 0:
            for i, waypoint in enumerate(self.waypoints):
                x, y, z = waypoint
                
                # 绘制空心圆圈（使用GLLinePlotItem绘制圆周）
                circle_points = []
                radius = 3.0  # 圆圈半径
                num_segments = 32  # 圆周分段数
                
                for j in range(num_segments + 1):
                    angle = 2 * np.pi * j / num_segments
                    # 在XY平面绘制圆圈
                    px = x + radius * np.cos(angle)
                    py = y + radius * np.sin(angle)
                    pz = z
                    circle_points.append([px, py, pz])
                
                # 绘制圆圈（橙色）
                circle_line = GLLinePlotItem(
                    pos=np.array(circle_points),
                    color=pg.mkColor(255, 165, 0, 200),  # 橙色，半透明
                    width=3,
                    antialias=True
                )
                self.view_3d.addItem(circle_line)
                self.waypoint_items.append(circle_line)
                
                # 添加检查点标签
                waypoint_text = GLTextItem(pos=(x, y, z + 8), text=f"检查点{i+1}", color='white')
                self.view_3d.addItem(waypoint_text)
                self.waypoint_items.append(waypoint_text)
        
        # 绘制完整巡检路径（起点→检查点→终点→检查点→起点）
        if len(self.inspection_path) > 1:
            for i in range(len(self.inspection_path) - 1):
                x1, y1, z1 = self.inspection_path[i]
                x2, y2, z2 = self.inspection_path[i + 1]
                
                # 去程用蓝色，返程用橙色
                if i < len(self.waypoints) + 1:  # 去程
                    path_color = pg.mkColor(0, 0, 255, 200)  # 蓝色
                else:  # 返程
                    path_color = pg.mkColor(255, 165, 0, 200)  # 橙色
                
                path_line = GLLinePlotItem(
                    pos=np.array([[x1, y1, z1], [x2, y2, z2]]),
                    color=path_color,
                    width=4
                )
                self.view_3d.addItem(path_line)
                self.path_line_items.append(path_line)
        
        # 绘制无人机
        drone_positions = []
        drone_colors = []
        drone_sizes = []
        
        for i, drone in enumerate(self.drones_data):
            x, y, z = drone.get('position', [0, 0, 0])
            drone_positions.append([x, y, z])
            
            status = drone.get('status', 'idle')
            if status == 'flying':
                drone_colors.append([0, 0.5, 1, 1])  # 蓝色
            elif status == 'returning':
                drone_colors.append([1, 0.65, 0, 1])  # 橙色
            else:
                drone_colors.append([0, 1, 0, 1])  # 绿色
            
            drone_sizes.append(40)
            
            # 添加文本标签（单行显示）
            battery = drone.get('battery', 100)
            text_item = GLTextItem(pos=(x, y, z + 8), text=f"D{i} {battery:.0f}%", color='white')
            self.view_3d.addItem(text_item)
            self.drone_text_items.append(text_item)
        
        if drone_positions:
            drone_scatter = GLScatterPlotItem(
                pos=np.array(drone_positions),
                color=np.array(drone_colors),
                size=drone_sizes,
                pxMode=False
            )
            self.view_3d.addItem(drone_scatter)
            self.drone_scatter_items.append(drone_scatter)
    
    def _draw_3d_formation_view(self):
        """绘制3D队形任务视图（给定起点和终点，无人机自主规划路径）"""
        # 绘制起点（绿色）
        if self.formation_start is not None:
            x, y, z = self.formation_start
            start_scatter = GLScatterPlotItem(
                pos=np.array([[x, y, z]]),
                color=np.array([[0, 1, 0, 1]]),  # 绿色
                size=[60],
                pxMode=False
            )
            self.view_3d.addItem(start_scatter)
            self.start_point_items.append(start_scatter)
            
            # 添加起点标签
            start_text = GLTextItem(pos=(x, y, z + 8), text="起点", color='white')
            self.view_3d.addItem(start_text)
            self.start_point_items.append(start_text)
        
        # 绘制终点（红色）
        if self.formation_end is not None:
            x, y, z = self.formation_end
            end_scatter = GLScatterPlotItem(
                pos=np.array([[x, y, z]]),
                color=np.array([[1, 0, 0, 1]]),  # 红色
                size=[60],
                pxMode=False
            )
            self.view_3d.addItem(end_scatter)
            self.end_point_items.append(end_scatter)
            
            # 添加终点标签
            end_text = GLTextItem(pos=(x, y, z + 8), text="终点", color='white')
            self.view_3d.addItem(end_text)
            self.end_point_items.append(end_text)
        
        # 绘制队形目标位置
        formation_positions = []
        formation_colors = []
        formation_sizes = []
        
        for i, target in enumerate(self.formation_targets):
            x, y, z = target
            formation_positions.append([x, y, z])
            
            # 领航机用金色，僚机用红色
            if i == self.leader_drone_idx:
                formation_colors.append([1, 0.84, 0, 0.7])  # 金色
                formation_sizes.append(60)
            else:
                formation_colors.append([1, 0, 0, 0.7])  # 红色
                formation_sizes.append(50)
            
            # 添加文本标签
            label = "领航机" if i == self.leader_drone_idx else f"僚机{i}"
            text_item = GLTextItem(pos=(x, y, z + 8), text=label, color='white')
            self.view_3d.addItem(text_item)
            self.formation_text_items.append(text_item)
        
        if formation_positions:
            formation_scatter = GLScatterPlotItem(
                pos=np.array(formation_positions),
                color=np.array(formation_colors),
                size=formation_sizes,
                pxMode=False
            )
            self.view_3d.addItem(formation_scatter)
            self.formation_target_items.append(formation_scatter)
        
        # 绘制队形连线（从领航机到僚机）
        if len(self.formation_targets) > 1:
            leader_pos = self.formation_targets[self.leader_drone_idx]
            for i in range(len(self.formation_targets)):
                if i != self.leader_drone_idx:
                    x1, y1, z1 = leader_pos
                    x2, y2, z2 = self.formation_targets[i]
                    
                    formation_line = GLLinePlotItem(
                        pos=np.array([[x1, y1, z1], [x2, y2, z2]]),
                        color=pg.mkColor(255, 215, 0, 150),  # 金色
                        width=6,
                        antialias=True
                    )
                    self.view_3d.addItem(formation_line)
                    self.path_line_items.append(formation_line)
        
        # 绘制无人机
        drone_positions = []
        drone_colors = []
        drone_sizes = []
        
        for i, drone in enumerate(self.drones_data):
            x, y, z = drone.get('position', [0, 0, 0])
            drone_positions.append([x, y, z])
            
            status = drone.get('status', 'idle')
            if status == 'flying':
                drone_colors.append([0, 0.5, 1, 1])  # 蓝色
            elif status == 'returning':
                drone_colors.append([1, 0.65, 0, 1])  # 橙色
            else:
                drone_colors.append([0, 1, 0, 1])  # 绿色
            
            drone_sizes.append(40)
            
            # 添加文本标签（单行显示）
            battery = drone.get('battery', 100)
            text_item = GLTextItem(pos=(x, y, z + 8), text=f"D{i} {battery:.0f}%", color='white')
            self.view_3d.addItem(text_item)
            self.drone_text_items.append(text_item)
        
        if drone_positions:
            drone_scatter = GLScatterPlotItem(
                pos=np.array(drone_positions),
                color=np.array(drone_colors),
                size=drone_sizes,
                pxMode=False
            )
            self.view_3d.addItem(drone_scatter)
            self.drone_scatter_items.append(drone_scatter)
        
        # 绘制无人机到目标位置的连线
        for i, drone in enumerate(self.drones_data):
            if i < len(self.formation_targets):
                drone_pos = drone.get('position', [0, 0, 0])
                target_pos = self.formation_targets[i]
                
                connection_line = GLLinePlotItem(
                    pos=np.array([drone_pos, target_pos]),
                    color=pg.mkColor(128, 128, 128, 150),  # 灰色虚线
                    width=2,
                    antialias=True
                )
                self.view_3d.addItem(connection_line)
                self.path_line_items.append(connection_line)
    
    def _draw_3d_encirclement_view(self):
        """绘制3D协同包围任务视图"""
        # 绘制目标位置（红色）
        if hasattr(self, 'target_position') and self.target_position is not None:
            x, y, z = self.target_position
            target_scatter = GLScatterPlotItem(
                pos=np.array([[x, y, z]]),
                color=np.array([[1, 0, 0, 1]]),  # 红色
                size=[60],
                pxMode=False
            )
            self.view_3d.addItem(target_scatter)
            self.start_point_items.append(target_scatter)
            
            # 添加目标标签
            target_text = GLTextItem(pos=(x, y, z + 8), text=f"目标\n{self.encirclement_time}步", color='white')
            self.view_3d.addItem(target_text)
            self.start_point_items.append(target_text)
            
            # 绘制包围半径球体（蓝色）
            if hasattr(self, 'encirclement_radius') and self.encirclement_radius > 0:
                # 使用多个点模拟球体
                num_points = 20
                for i in range(num_points):
                    angle = i * 2 * np.pi / num_points
                    radius_x = x + self.encirclement_radius * np.cos(angle)
                    radius_y = y + self.encirclement_radius * np.sin(angle)
                    radius_z = z
                    
                    radius_scatter = GLScatterPlotItem(
                        pos=np.array([[radius_x, radius_y, radius_z]]),
                        color=np.array([[0, 0.5, 1, 0.3]]),  # 半透明蓝色
                        size=[20],
                        pxMode=False
                    )
                    self.view_3d.addItem(radius_scatter)
                    self.start_point_items.append(radius_scatter)
            
            # 绘制包围成功提示
            if hasattr(self, 'encirclement_success') and self.encirclement_success:
                success_text = GLTextItem(pos=(x, y, z + 20), text="包围成功！", color='green')
                self.view_3d.addItem(success_text)
                self.start_point_items.append(success_text)
        
        # 绘制无人机
        drone_positions = []
        drone_colors = []
        drone_sizes = []
        
        for i, drone in enumerate(self.drones_data):
            x, y, z = drone.get('position', [0, 0, 0])
            drone_positions.append([x, y, z])
            
            status = drone.get('status', 'idle')
            if status == 'flying':
                drone_colors.append([0, 0.5, 1, 1])  # 蓝色
            else:
                drone_colors.append([0, 1, 0, 1])  # 绿色
            
            drone_sizes.append(40)
            
            # 添加文本标签
            battery = drone.get('battery', 100)
            text_item = GLTextItem(pos=(x, y, z + 8), text=f"D{i} {battery:.0f}%", color='white')
            self.view_3d.addItem(text_item)
            self.drone_text_items.append(text_item)
        
        if drone_positions:
            drone_scatter = GLScatterPlotItem(
                pos=np.array(drone_positions),
                color=np.array(drone_colors),
                size=drone_sizes,
                pxMode=False
            )
            self.view_3d.addItem(drone_scatter)
            self.drone_scatter_items.append(drone_scatter)
    
    def update_task_info(self, task_info):
        """更新任务信息"""
        if self.task_type == 'encirclement':
            encirclement_time = task_info.get('encirclement_time', 0)
            encirclement_success = task_info.get('encirclement_success', False)
            if encirclement_success:
                self.task_info_label.setText(f"协同包围任务：包围成功！持续{encirclement_time}步")
            else:
                self.task_info_label.setText(f"协同包围任务：包围中...{encirclement_time}/50 步")
        else:  # formation
            formation_error = task_info.get('formation_error', 0.0)
            self.task_info_label.setText(f"队形任务：队形误差 {formation_error:.2f} 米")
