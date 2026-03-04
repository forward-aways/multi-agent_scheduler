"""
多智能体协作任务调度系统 - UI包

该包包含多智能体系统的所有UI组件，
包括主窗口、可视化组件和后端集成。

模块:
- main_window: 主应用程序窗口和菜单系统
- server_visualization: 服务器调度可视化组件
- drone_visualization: 无人机调度可视化组件
- logistics_visualization: 物流调度可视化组件
- backend_controller: 与多智能体调度后端的集成
"""

# 导入主类以便轻松访问
from .main_window import MainWindow
from .server_visualization import ServerVisualizationWidget
from .drone_visualization import DroneMissionVisualizationWidget
from .logistics_visualization import LogisticsVisualizationWidget
from .backend_controller import BackendController

__all__ = [
    'MainWindow',
    'ServerVisualizationWidget',
    'DroneMissionVisualizationWidget',
    'LogisticsVisualizationWidget',
    'BackendController'
]

__version__ = "1.0.0"
__author__ = "多智能体任务调度系统"
