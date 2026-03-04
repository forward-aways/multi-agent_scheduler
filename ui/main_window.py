"""
多智能体协作任务调度系统 - GUI应用程序
主窗口和菜单栏实现
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QMenuBar, QStatusBar, QLabel, QToolBar,
    QMessageBox, QSplitter, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QFont, QAction
from utils.logging_config import ProjectLogger

ui_logger = ProjectLogger('ui', log_dir='logs')


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多智能体协作任务调度系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置窗口图标（如果有的话）
        # self.setWindowIcon(QIcon('icon.png'))
        
        # 初始化后端控制器
        from .backend_controller import BackendController
        self.backend_controller = BackendController()
        
        # 初始化界面组件
        self.init_ui()
        self.create_menu_bar()
        self.create_tool_bar()
        self.create_status_bar()
        
        # 连接信号
        self.backend_controller.server_data_updated.connect(self.on_server_data_updated)
        self.backend_controller.drone_data_updated.connect(self.on_drone_data_updated)
        self.backend_controller.logistics_data_updated.connect(self.on_logistics_data_updated)
        self.backend_controller.training_status_updated.connect(self.on_training_status_updated)
        self.backend_controller.inference_status_updated.connect(self.on_inference_status_updated)
        self.backend_controller.system_error.connect(self.on_system_error)
        
        # 连接后端数据更新信号到可视化组件
        self.backend_controller.server_data_updated.connect(self.server_tab.update_from_backend)
        self.backend_controller.drone_data_updated.connect(self.drone_tab.update_from_backend)
        self.backend_controller.logistics_data_updated.connect(self.logistics_tab.update_from_backend)
        
        # 默认最大化窗口
        self.showMaximized()
        
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标题标签
        title_label = QLabel("多智能体协作任务调度系统")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_layout.addWidget(title_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 创建左侧控制面板
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # 创建右侧选项卡控件
        tab_widget = QTabWidget()
        
        # 添加各个调度场景的选项卡
        from ui.server_visualization import ServerVisualizationWidget
        from ui.drone_visualization import DroneMissionVisualizationWidget
        from ui.logistics_visualization import LogisticsVisualizationWidget
        from ui.evaluation_widget import EvaluationWidget
        
        self.server_tab = ServerVisualizationWidget()
        self.server_tab.backend_controller = self.backend_controller
        self.drone_tab = DroneMissionVisualizationWidget()
        self.drone_tab.backend_controller = self.backend_controller
        self.logistics_tab = LogisticsVisualizationWidget()
        self.logistics_tab.backend_controller = self.backend_controller
        self.evaluation_tab = EvaluationWidget()
        
        tab_widget.addTab(self.server_tab, "服务器调度")
        tab_widget.addTab(self.drone_tab, "无人机调度")
        tab_widget.addTab(self.logistics_tab, "物流调度")
        tab_widget.addTab(self.evaluation_tab, "自动化评估")
        
        # 连接标签页切换信号
        tab_widget.currentChanged.connect(self.on_tab_changed)
        
        splitter.addWidget(tab_widget)
        splitter.setSizes([300, 900])  # 设置分割器大小比例
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QGroupBox("控制面板")
        layout = QVBoxLayout(panel)
        
        # 添加控制按钮
        from PyQt6.QtWidgets import QPushButton
        
        btn_start = QPushButton("启动调度")
        btn_start.setStyleSheet("background-color: lightgreen; padding: 10px; border: 1px solid gray;")
        btn_start.clicked.connect(self.start_scheduling)
        btn_stop = QPushButton("停止调度")
        btn_stop.setStyleSheet("background-color: lightcoral; padding: 10px; border: 1px solid gray;")
        btn_stop.clicked.connect(self.stop_scheduling)
        btn_reset = QPushButton("重置系统")
        btn_reset.setStyleSheet("background-color: lightyellow; padding: 10px; border: 1px solid gray;")
        btn_reset.clicked.connect(self.reset_system)
        
        layout.addWidget(btn_start)
        layout.addWidget(btn_stop)
        layout.addWidget(btn_reset)
        
        # 添加配置信息
        config_group = QGroupBox("系统配置")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("服务器数量:"), 0, 0)
        config_layout.addWidget(QLabel("5"), 0, 1)
        config_layout.addWidget(QLabel("无人机数量:"), 1, 0)
        config_layout.addWidget(QLabel("3"), 1, 1)
        config_layout.addWidget(QLabel("物流节点:"), 2, 0)
        config_layout.addWidget(QLabel("3"), 2, 1)
        
        layout.addWidget(config_group)
        
        # 添加调度模式选择
        mode_group = QGroupBox("调度模式")
        mode_layout = QVBoxLayout(mode_group)
        
        from PyQt6.QtWidgets import QRadioButton
        self.auto_mode_radio = QRadioButton("自动调度模式")
        self.auto_mode_radio.setChecked(False)  # 默认手动模式
        self.manual_mode_radio = QRadioButton("手动调度模式")
        self.manual_mode_radio.setChecked(True)  # 默认手动模式
        
        mode_layout.addWidget(self.auto_mode_radio)
        mode_layout.addWidget(self.manual_mode_radio)
        
        # 连接模式切换事件
        self.auto_mode_radio.toggled.connect(self.on_mode_changed)
        self.manual_mode_radio.toggled.connect(self.on_mode_changed)
        
        # 在初始化时也要设置后端控制器的模式
        self.backend_controller.set_manual_auto_mode('manual')  # 初始化时设置为手动模式
        
        layout.addWidget(mode_group)
        
        # 添加算法选择
        algorithm_group = QGroupBox("算法选择")
        algorithm_layout = QVBoxLayout(algorithm_group)
        
        from PyQt6.QtWidgets import QComboBox
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['maddpg', 'mappo'])
        self.algorithm_combo.setCurrentText('mappo')  # 默认使用MAPPO
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        
        algorithm_layout.addWidget(QLabel("选择强化学习算法:"))
        algorithm_layout.addWidget(self.algorithm_combo)
        
        # 添加算法说明
        self.algorithm_info_label = QLabel("MAPPO: 多智能体近端策略优化算法")
        self.algorithm_info_label.setStyleSheet("color: gray; font-size: 10px;")
        self.algorithm_info_label.setWordWrap(True)
        algorithm_layout.addWidget(self.algorithm_info_label)
        
        layout.addWidget(algorithm_group)
        
        # 添加实时状态
        status_group = QGroupBox("实时状态")
        status_layout = QVBoxLayout(status_group)
        
        self.status_labels = {
            'servers': QLabel("服务器状态: 正常"),
            'drones': QLabel("无人机状态: 正常"),
            'logistics': QLabel("物流状态: 正常"),
            'tasks_completed': QLabel("完成任务数: 0"),
            'tasks_failed': QLabel("失败任务数: 0")
        }
        
        for label in self.status_labels.values():
            status_layout.addWidget(label)
        
        layout.addWidget(status_group)
        layout.addStretch()  # 添加弹性空间
        
        return panel
        
    def create_server_tab(self):
        """创建服务器调度选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 添加服务器调度的可视化组件
        title = QLabel("服务器调度 - 任务分配和资源管理")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # 服务器状态网格
        servers_grid = QGroupBox("服务器状态")
        grid_layout = QGridLayout(servers_grid)
        
        # 创建5个服务器的显示
        for i in range(5):
            server_box = QGroupBox(f"服务器 {i+1}")
            server_layout = QVBoxLayout(server_box)
            
            cpu_label = QLabel(f"CPU: 0%")
            mem_label = QLabel(f"内存: 0%")
            task_label = QLabel(f"任务数: 0")
            status_label = QLabel(f"状态: 空闲")
            
            server_layout.addWidget(cpu_label)
            server_layout.addWidget(mem_label)
            server_layout.addWidget(task_label)
            server_layout.addWidget(status_label)
            
            grid_layout.addWidget(server_box, i // 3, i % 3)  # 每行最多3个
        
        layout.addWidget(servers_grid)
        
        # 任务队列显示
        task_queue_group = QGroupBox("任务队列")
        task_layout = QVBoxLayout(task_queue_group)
        task_queue_label = QLabel("等待处理的任务将在这里显示...")
        task_layout.addWidget(task_queue_label)
        layout.addWidget(task_queue_group)
        
        layout.addStretch()
        return widget
        
    def create_drone_tab(self):
        """创建无人机调度选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 添加无人机调度的可视化组件
        title = QLabel("无人机调度 - 路径规划和任务执行")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # 无人机状态网格
        drones_grid = QGroupBox("无人机状态")
        grid_layout = QGridLayout(drones_grid)
        
        # 创建3个无人机的显示
        for i in range(3):
            drone_box = QGroupBox(f"无人机 {i+1}")
            drone_layout = QVBoxLayout(drone_box)
            
            pos_label = QLabel(f"位置: [0, 0, 0]")
            battery_label = QLabel(f"电量: 100%")
            task_label = QLabel(f"任务: 无")
            status_label = QLabel(f"状态: 待命")
            
            drone_layout.addWidget(pos_label)
            drone_layout.addWidget(battery_label)
            drone_layout.addWidget(task_label)
            drone_layout.addWidget(status_label)
            
            grid_layout.addWidget(drone_box, i // 3, i % 3)  # 每行最多3个
        
        layout.addWidget(drones_grid)
        
        # 地图可视化区域（简单表示）
        map_group = QGroupBox("任务地图")
        map_layout = QVBoxLayout(map_group)
        map_label = QLabel("2D/3D地图可视化将在这里显示...")
        map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        map_label.setStyleSheet("border: 1px solid gray; background-color: lightblue; padding: 50px;")
        map_layout.addWidget(map_label)
        layout.addWidget(map_group)
        
        layout.addStretch()
        return widget
        
    def create_logistics_tab(self):
        """创建物流调度选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 添加物流调度的可视化组件
        title = QLabel("物流调度 - 库存管理和配送调度")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # 物流节点状态网格
        nodes_grid = QGroupBox("物流节点状态")
        grid_layout = QGridLayout(nodes_grid)
        
        # 创建3个物流节点的显示
        for i in range(3):
            node_box = QGroupBox(f"节点 {i+1}")
            node_layout = QVBoxLayout(node_box)
            
            inventory_label = QLabel(f"库存: A:{100}, B:{100}, C:{100}")
            orders_label = QLabel(f"订单数: 0")
            vehicles_label = QLabel(f"车辆数: 0")
            status_label = QLabel(f"状态: 正常")
            
            node_layout.addWidget(inventory_label)
            node_layout.addWidget(orders_label)
            node_layout.addWidget(vehicles_label)
            node_layout.addWidget(status_label)
            
            grid_layout.addWidget(node_box, i // 3, i % 3)  # 每行最多3个
        
        layout.addWidget(nodes_grid)
        
        # 订单管理区域
        orders_group = QGroupBox("订单管理")
        orders_layout = QVBoxLayout(orders_group)
        orders_label = QLabel("当前订单信息将在这里显示...")
        orders_layout.addWidget(orders_label)
        layout.addWidget(orders_group)
        
        layout.addStretch()
        return widget
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        new_action = QAction('新建', self)
        new_action.setShortcut('Ctrl+N')
        file_menu.addAction(new_action)
        
        open_action = QAction('打开', self)
        open_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_action)
        
        save_action = QAction('保存', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu('编辑')
        
        config_action = QAction('系统配置', self)
        config_action.setShortcut('Ctrl+,')
        edit_menu.addAction(config_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        refresh_action = QAction('刷新', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.refresh_view)
        view_menu.addAction(refresh_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        train_action = QAction('训练模型', self)
        train_action.triggered.connect(self.train_model)
        tools_menu.addAction(train_action)
        
        infer_action = QAction('推理模式', self)
        infer_action.triggered.connect(self.infer_model)
        tools_menu.addAction(infer_action)
        
        tools_menu.addSeparator()
        
        eval_action = QAction('自动化评估', self)
        eval_action.setShortcut('Ctrl+E')
        eval_action.triggered.connect(self.open_evaluation)
        tools_menu.addAction(eval_action)
        
        tools_menu.addSeparator()
        
        # API网页入口
        api_action = QAction('API控制台', self)
        api_action.setShortcut('Ctrl+A')
        api_action.triggered.connect(self.open_api_web)
        tools_menu.addAction(api_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = self.addToolBar('工具栏')
        
        # 添加常用工具按钮
        start_action = QAction('启动', self)
        start_action.triggered.connect(self.start_scheduling)
        toolbar.addAction(start_action)
        
        stop_action = QAction('停止', self)
        stop_action.triggered.connect(self.stop_scheduling)
        toolbar.addAction(stop_action)
        
        reset_action = QAction('重置', self)
        reset_action.triggered.connect(self.reset_system)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        train_action = QAction('训练', self)
        train_action.triggered.connect(self.train_model)
        toolbar.addAction(train_action)
        
        infer_action = QAction('推理', self)
        infer_action.triggered.connect(self.infer_model)
        toolbar.addAction(infer_action)
        
        toolbar.addSeparator()
        
        eval_action = QAction('评估', self)
        eval_action.triggered.connect(self.open_evaluation)
        toolbar.addAction(eval_action)
        
        toolbar.addSeparator()
        
        # API网页入口按钮
        api_action = QAction('API', self)
        api_action.triggered.connect(self.open_api_web)
        toolbar.addAction(api_action)
        
    def create_status_bar(self):
        """创建状态栏"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # 添加状态信息
        status_bar.showMessage('就绪')
        
        # 添加永久信息
        self.status_label = QLabel('多智能体协作任务调度系统 v1.0')
        status_bar.addPermanentWidget(self.status_label)
        
    def refresh_view(self):
        """刷新视图"""
        # 这里可以添加刷新逻辑
        self.statusBar().showMessage('视图已刷新', 2000)
        
    def start_scheduling(self):
        """启动调度"""
        # 通知后端开始调度
        self.backend_controller.start_inference()
        self.statusBar().showMessage('调度已启动', 2000)
        
    def stop_scheduling(self):
        """停止调度"""
        # 通知后端停止调度
        self.backend_controller.stop_current_operation()
        self.statusBar().showMessage('调度已停止', 2000)
        
    def reset_system(self):
        """重置系统"""
        reply = QMessageBox.question(self, '重置系统', '确定要重置系统吗？',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # 通知后端重置系统
            self.backend_controller.reset_system()
            self.statusBar().showMessage('系统已重置', 2000)
        
    def train_model(self):
        """训练模型"""
        self.backend_controller.start_training(50)  # 训练50个episode
        self.statusBar().showMessage('开始训练模型...', 2000)
        
    def infer_model(self):
        """推理模式"""
        self.backend_controller.start_inference()
        self.statusBar().showMessage('开始推理...', 2000)
        
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, '关于', 
                         '多智能体协作任务调度系统\n版本 1.0\n\n用于演示多智能体协作调度算法的图形界面。')

    # 信号处理方法
    def on_server_data_updated(self, data):
        """处理服务器数据更新"""
        ui_logger.debug("收到服务器数据更新")
        
        # 更新实时状态
        if data and len(data) > 0:
            # 更新服务器状态
            active_servers = sum(1 for server in data if server.get('status') != 'idle')
            server_status = f"活跃服务器: {active_servers}/{len(data)}"
            self.status_labels['servers'].setText(server_status)
            
            # 更新任务统计
            if 'tasks_completed' in data[0]:
                self.status_labels['tasks_completed'].setText(f"完成任务数: {data[0]['tasks_completed']}")
            if 'tasks_failed' in data[0]:
                self.status_labels['tasks_failed'].setText(f"失败任务数: {data[0]['tasks_failed']}")
    
    def on_drone_data_updated(self, data):
        """处理无人机数据更新"""
        ui_logger.debug("收到无人机数据更新")
        
        # 更新无人机状态
        if data and 'drones' in data and len(data['drones']) > 0:
            drones = data['drones']
            active_drones = sum(1 for drone in drones if drone.get('status') in ['flying', 'hovering'])
            drone_status = f"活跃无人机: {active_drones}/{len(drones)}"
            self.status_labels['drones'].setText(drone_status)
    
    def on_logistics_data_updated(self, data):
        """处理物流数据更新"""
        ui_logger.debug("收到物流数据更新")
        
        # 更新物流状态
        if data and len(data) > 0:
            normal_nodes = sum(1 for node in data if node.get('status') == 'normal')
            logistics_status = f"正常节点: {normal_nodes}/{len(data)}"
            self.status_labels['logistics'].setText(logistics_status)
    
    def on_tab_changed(self, index):
        """处理标签页切换"""
        tab_names = ['server', 'drone', 'logistics']
        if 0 <= index < len(tab_names):
            mode = tab_names[index]
            self.backend_controller.set_mode(mode)
            self.statusBar().showMessage(f'切换到{["服务器", "无人机", "物流"][index]}模式', 2000)
    
    def on_training_status_updated(self, status, avg_reward):
        """处理训练状态更新"""
        self.statusBar().showMessage(f"{status} (平均奖励: {avg_reward:.2f})", 3000)
    
    def on_inference_status_updated(self, status):
        """处理推理状态更新"""
        self.statusBar().showMessage(status, 3000)
    
    def on_mode_changed(self, checked=None):
        """处理调度模式改变"""
        ui_logger.debug(f"on_mode_changed called with checked={checked}")
        if self.auto_mode_radio.isChecked():
            mode = 'auto'
        else:  # 手动模式被选中或两个都没被选中
            mode = 'manual'
        
        ui_logger.debug(f"Determined mode = {mode}")
        # 通知后端控制器切换模式
        self.backend_controller.set_manual_auto_mode(mode)
        self.statusBar().showMessage(f'调度模式已切换为: {mode}', 2000)
    
    def on_algorithm_changed(self, algorithm):
        """处理算法选择改变"""
        ui_logger.info(f"算法已切换为: {algorithm}")
        
        # 更新算法说明
        if algorithm == 'maddpg':
            self.algorithm_info_label.setText("MADDPG: 多智能体深度确定性策略梯度算法")
        elif algorithm == 'mappo':
            self.algorithm_info_label.setText("MAPPO: 多智能体近端策略优化算法")
        
        # 通知后端控制器切换算法
        self.backend_controller.set_algorithm(algorithm)
        self.statusBar().showMessage(f'算法已切换为: {algorithm}', 2000)
    
    def on_system_error(self, error_msg):
        """处理系统错误"""
        QMessageBox.critical(self, "系统错误", error_msg)
    
    def open_evaluation(self):
        """打开自动化评估界面"""
        # 切换到评估标签页
        self.findChild(QTabWidget).setCurrentWidget(self.evaluation_tab)
        self.statusBar().showMessage('已切换到自动化评估界面', 2000)
    
    def open_api_web(self):
        """打开API网页控制台"""
        import webbrowser
        import socket
        
        api_url = 'http://127.0.0.1:5003/'
        
        # 检查API服务器是否已在运行
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 5003))
            sock.close()
            
            if result != 0:
                QMessageBox.information(self, 'API服务器', 'API服务器正在启动中，请稍后再试')
                return
        except:
            pass
        
        # 打开浏览器
        webbrowser.open(api_url)
        self.statusBar().showMessage(f'已打开API控制台: {api_url}', 3000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())