"""
Multi-Agent Collaborative Task Scheduling System - UI Launcher
启动时自动启动内置API服务器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal


class APIServerThread(QThread):
    """API服务器线程"""
    server_started = pyqtSignal()
    
    def __init__(self, port=5003):
        super().__init__()
        self.port = port
        self.running = False
        
    def run(self):
        """在线程中运行API服务器"""
        try:
            from flask import Flask, jsonify, request, send_from_directory
            from api.scheduler_api import SchedulerAPI
            
            app = Flask(__name__, static_folder='templates')
            api = SchedulerAPI()
            
            # 提供网页界面
            @app.route('/')
            def index():
                return send_from_directory('templates', 'index.html')
            
            # API端点
            @app.route('/api/status')
            def get_status():
                return jsonify(api.get_engine_status())
            
            @app.route('/api/plugins')
            def get_plugins():
                category = request.args.get('category')
                return jsonify(api.get_plugin_list(category))
            
            @app.route('/api/scenarios')
            def get_scenarios():
                return jsonify(api.get_available_scenarios())
            
            @app.route('/api/strategy/current')
            def get_current_strategy():
                return jsonify(api.get_current_strategy())
            
            @app.route('/api/scenario/load', methods=['POST'])
            def load_scenario():
                config = request.get_json()
                return jsonify(api.load_scenario(config))
            
            @app.route('/api/strategy/switch', methods=['POST'])
            def switch_strategy():
                data = request.get_json()
                strategy_name = data.get('strategy_name')
                config = data.get('config', {})
                return jsonify(api.switch_strategy(strategy_name, config))
            
            @app.route('/api/task/allocate', methods=['POST'])
            def allocate_task():
                data = request.get_json()
                task_info = data.get('task_info', {})
                agent_id = data.get('agent_id')
                return jsonify(api.request_task_allocation(task_info, agent_id))
            
            # 初始化API
            print("[API] 正在初始化...")
            api.initialize()
            print("[API] ✅ 初始化成功")
            
            self.running = True
            self.server_started.emit()
            
            print(f"[API] 启动服务器: http://127.0.0.1:{self.port}/")
            app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True, use_reloader=False)
            
        except Exception as e:
            print(f"[API] 错误: {e}")
            print("[API] 请确保已安装Flask: pip install flask")


def main():
    """UI应用程序入口点"""
    try:
        print("=" * 60)
        print("多智能体任务调度系统")
        print("=" * 60)
        
        # 启动API服务器线程
        print("\n[1/2] 正在启动API服务器...")
        api_thread = APIServerThread(port=5003)
        api_thread.start()
        
        # 等待API服务器启动
        time.sleep(2)
        print("[API] ✅ API服务器已启动")
        print(f"[API] 网页控制台: http://127.0.0.1:5003/")
        
        # 创建Qt应用
        print("\n[2/2] 正在启动UI界面...")
        app = QApplication(sys.argv)
        
        # 导入并创建主窗口
        from ui.main_window import MainWindow
        window = MainWindow()
        window.show()
        
        print("[UI] ✅ UI界面已启动\n")
        
        # 运行应用程序
        sys.exit(app.exec())
        
    except ImportError as e:
        print(f"错误：无法启动UI界面，请确保已安装PyQt6")
        print(f"错误详情：{e}")
        print("可通过以下命令安装：pip install PyQt6")
        sys.exit(1)


if __name__ == '__main__':
    main()
