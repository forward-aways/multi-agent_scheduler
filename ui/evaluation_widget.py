"""
评估模块UI组件
提供自动化评估的配置、执行和报告展示功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QGroupBox, QFormLayout,
    QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QFileDialog, QMessageBox,
    QTabWidget, QSplitter, QCheckBox, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import json
from datetime import datetime
from pathlib import Path

from api.evaluation_api import EvaluationAPI
from utils.logging_config import ProjectLogger

ui_logger = ProjectLogger('ui', log_dir='logs')


class EvaluationWorker(QThread):
    """评估工作线程"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, int, str)  # 当前进度, 总进度, 状态信息
    episode_completed = pyqtSignal(int, dict)  # 回合数, 回合数据
    evaluation_finished = pyqtSignal(dict)  # 评估结果
    error_occurred = pyqtSignal(str)  # 错误信息
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.eval_api = EvaluationAPI()
        self.is_running = False
    
    def run(self):
        """运行评估"""
        self.is_running = True
        
        try:
            # 设置回调以获取进度
            episode_count = [0]
            
            def on_episode_end(data):
                episode_count[0] += 1
                self.episode_completed.emit(episode_count[0], data)
                self.progress_updated.emit(
                    episode_count[0],
                    self.config.get('episodes', 100),
                    f"完成回合 {episode_count[0]}/{self.config.get('episodes', 100)}"
                )
            
            # 运行评估
            result = self.eval_api.run_evaluation(self.config)
            
            if self.is_running:
                self.evaluation_finished.emit(result)
                
        except Exception as e:
            if self.is_running:
                self.error_occurred.emit(str(e))
    
    def stop(self):
        """停止评估"""
        self.is_running = False
        self.wait(1000)  # 等待1秒


class EvaluationConfigDialog(QDialog):
    """评估配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("评估配置")
        self.setMinimumWidth(500)
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # 基本信息
        basic_group = QGroupBox("基本信息")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入评估名称")
        self.name_edit.setText(f"评估_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        basic_layout.addRow("评估名称:", self.name_edit)
        
        layout.addWidget(basic_group)
        
        # 环境配置
        env_group = QGroupBox("环境配置")
        env_layout = QFormLayout(env_group)
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "server_environment",
            "drone_environment",
            "logistics_environment"
        ])
        env_layout.addRow("环境类型:", self.env_combo)
        
        # 环境参数
        self.num_agents_spin = QSpinBox()
        self.num_agents_spin.setRange(1, 20)
        self.num_agents_spin.setValue(3)
        env_layout.addRow("智能体数量:", self.num_agents_spin)
        
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 10000)
        self.max_steps_spin.setValue(200)
        self.max_steps_spin.setSingleStep(50)
        env_layout.addRow("最大步数:", self.max_steps_spin)
        
        layout.addWidget(env_group)
        
        # 策略配置
        strategy_group = QGroupBox("策略配置")
        strategy_layout = QFormLayout(strategy_group)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "random_strategy",
            "mappo_strategy"
        ])
        strategy_layout.addRow("策略类型:", self.strategy_combo)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("模型路径（可选）")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self.browse_model_path)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(btn_browse)
        strategy_layout.addRow("模型路径:", path_layout)
        
        layout.addWidget(strategy_group)
        
        # 评估参数
        eval_group = QGroupBox("评估参数")
        eval_layout = QFormLayout(eval_group)
        
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 1000)
        self.episodes_spin.setValue(50)
        self.episodes_spin.setSingleStep(10)
        eval_layout.addRow("评估回合数:", self.episodes_spin)
        
        layout.addWidget(eval_group)
        
        # 导出配置
        export_group = QGroupBox("导出配置")
        export_layout = QFormLayout(export_group)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["json", "csv"])
        export_layout.addRow("导出格式:", self.export_format_combo)
        
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setText("evaluations/ui_evaluation")
        btn_export_browse = QPushButton("浏览...")
        btn_export_browse.clicked.connect(self.browse_export_path)
        
        export_path_layout = QHBoxLayout()
        export_path_layout.addWidget(self.export_path_edit)
        export_path_layout.addWidget(btn_export_browse)
        export_layout.addRow("导出路径:", export_path_layout)
        
        layout.addWidget(export_group)
        
        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def browse_model_path(self):
        """浏览模型路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.model_path_edit.setText(path)
    
    def browse_export_path(self):
        """浏览导出路径"""
        path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if path:
            self.export_path_edit.setText(path)
    
    def get_config(self) -> dict:
        """获取配置"""
        return {
            'name': self.name_edit.text(),
            'environment': self.env_combo.currentText(),
            'strategy': self.strategy_combo.currentText(),
            'episodes': self.episodes_spin.value(),
            'max_steps': self.max_steps_spin.value(),
            'env_config': {
                'num_servers': self.num_agents_spin.value(),
                'num_drones': self.num_agents_spin.value(),
                'num_vehicles': self.num_agents_spin.value(),
                'max_steps': self.max_steps_spin.value()
            },
            'strategy_config': {
                'model_path': self.model_path_edit.text() if self.model_path_edit.text() else None,
                'num_actions': 27
            },
            'export_format': self.export_format_combo.currentText(),
            'export_path': self.export_path_edit.text()
        }


class EvaluationReportWidget(QWidget):
    """评估报告展示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 概览标签页
        self.overview_widget = QWidget()
        self.init_overview_tab()
        self.tab_widget.addTab(self.overview_widget, "概览")
        
        # 指标标签页
        self.metrics_widget = QWidget()
        self.init_metrics_tab()
        self.tab_widget.addTab(self.metrics_widget, "详细指标")
        
        # 回合数据标签页
        self.episodes_widget = QWidget()
        self.init_episodes_tab()
        self.tab_widget.addTab(self.episodes_widget, "回合数据")
        
        # 原始数据标签页
        self.raw_data_widget = QTextEdit()
        self.raw_data_widget.setReadOnly(True)
        self.tab_widget.addTab(self.raw_data_widget, "原始数据")
        
        layout.addWidget(self.tab_widget)
        
        # 导出按钮
        btn_layout = QHBoxLayout()
        
        self.btn_export_json = QPushButton("导出JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        btn_layout.addWidget(self.btn_export_json)
        
        self.btn_export_csv = QPushButton("导出CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        btn_layout.addWidget(self.btn_export_csv)
        
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        # 当前报告数据
        self.current_report = None
    
    def init_overview_tab(self):
        """初始化概览标签页"""
        layout = QVBoxLayout(self.overview_widget)
        
        # 基本信息
        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        layout.addWidget(self.overview_text)
    
    def init_metrics_tab(self):
        """初始化指标标签页"""
        layout = QVBoxLayout(self.metrics_widget)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.metrics_table)
    
    def init_episodes_tab(self):
        """初始化回合数据标签页"""
        layout = QVBoxLayout(self.episodes_widget)
        
        self.episodes_table = QTableWidget()
        self.episodes_table.setColumnCount(4)
        self.episodes_table.setHorizontalHeaderLabels(["回合", "步数", "总奖励", "平均奖励"])
        self.episodes_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.episodes_table)
    
    def display_report(self, report: dict):
        """显示评估报告"""
        self.current_report = report
        
        if not report or not report.get('success'):
            self.overview_text.setText("评估失败或暂无数据")
            return
        
        evaluation = report.get('evaluation', {})
        metrics = evaluation.get('metrics', {})
        
        # 更新概览
        overview_text = f"""
<h2>评估报告: {evaluation.get('name', 'Unknown')}</h2>

<h3>基本信息</h3>
<ul>
<li>评估回合数: {evaluation.get('episodes', 0)}</li>
<li>运行时长: {evaluation.get('duration', 0):.2f} 秒</li>
</ul>

<h3>关键指标</h3>
<ul>
<li>成功率: {metrics.get('success_rate', 0):.2%}</li>
<li>总奖励: {metrics.get('total_reward', 0):.2f}</li>
<li>平均奖励/步: {metrics.get('avg_reward_per_step', 0):.4f}</li>
<li>任务完成率: {metrics.get('task_completion_rate', 0):.2%}</li>
<li>平均延迟: {metrics.get('avg_delay', 0):.2f}</li>
<li>负载均衡度: {metrics.get('load_balance_score', 0):.4f}</li>
<li>稳定性得分: {metrics.get('stability_score', 0):.4f}</li>
</ul>
"""
        self.overview_text.setHtml(overview_text)
        
        # 更新指标表格
        self.metrics_table.setRowCount(0)
        metric_items = [
            ("成功率", f"{metrics.get('success_rate', 0):.2%}"),
            ("总奖励", f"{metrics.get('total_reward', 0):.2f}"),
            ("平均奖励/步", f"{metrics.get('avg_reward_per_step', 0):.4f}"),
            ("奖励方差", f"{metrics.get('reward_variance', 0):.4f}"),
            ("任务完成率", f"{metrics.get('task_completion_rate', 0):.2%}"),
            ("总任务数", str(metrics.get('total_tasks', 0))),
            ("已完成任务", str(metrics.get('completed_tasks', 0))),
            ("失败任务", str(metrics.get('failed_tasks', 0))),
            ("平均延迟", f"{metrics.get('avg_delay', 0):.2f}"),
            ("最大延迟", f"{metrics.get('max_delay', 0):.2f}"),
            ("最小延迟", f"{metrics.get('min_delay', 0):.2f}"),
            ("延迟标准差", f"{metrics.get('delay_std', 0):.2f}"),
            ("负载均衡度", f"{metrics.get('load_balance_score', 0):.4f}"),
            ("平均资源利用率", f"{metrics.get('avg_resource_usage', 0):.2f}"),
            ("资源效率", f"{metrics.get('resource_efficiency', 0):.4f}"),
            ("收敛回合", str(metrics.get('convergence_episode', -1))),
            ("稳定性得分", f"{metrics.get('stability_score', 0):.4f}"),
        ]
        
        for i, (key, value) in enumerate(metric_items):
            self.metrics_table.insertRow(i)
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
        
        # 更新回合数据表格
        raw_data = report.get('evaluation', {}).get('raw_data', [])
        self.episodes_table.setRowCount(len(raw_data))
        
        for i, episode_data in enumerate(raw_data):
            self.episodes_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.episodes_table.setItem(i, 1, QTableWidgetItem(str(episode_data.get('steps', 0))))
            self.episodes_table.setItem(i, 2, QTableWidgetItem(f"{episode_data.get('total_reward', 0):.2f}"))
            
            avg_reward = episode_data.get('total_reward', 0) / max(1, episode_data.get('steps', 1))
            self.episodes_table.setItem(i, 3, QTableWidgetItem(f"{avg_reward:.4f}"))
        
        # 更新原始数据
        self.raw_data_widget.setText(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    
    def export_json(self):
        """导出为JSON"""
        if not self.current_report:
            QMessageBox.warning(self, "警告", "没有可导出的报告")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "导出JSON", "evaluation_report.json", "JSON Files (*.json)"
        )
        
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.current_report, f, indent=2, ensure_ascii=False, default=str)
            QMessageBox.information(self, "成功", f"报告已导出到:\n{path}")
    
    def export_csv(self):
        """导出为CSV"""
        if not self.current_report:
            QMessageBox.warning(self, "警告", "没有可导出的报告")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "导出CSV", "evaluation_report.csv", "CSV Files (*.csv)"
        )
        
        if path:
            import csv
            raw_data = self.current_report.get('evaluation', {}).get('raw_data', [])
            
            with open(path, 'w', newline='', encoding='utf-8') as f:
                if raw_data:
                    writer = csv.DictWriter(f, fieldnames=raw_data[0].keys())
                    writer.writeheader()
                    writer.writerows(raw_data)
            
            QMessageBox.information(self, "成功", f"报告已导出到:\n{path}")


class EvaluationWidget(QWidget):
    """评估模块主组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.evaluation_worker = None
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("自动化评估模块")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 控制区域
        control_group = QGroupBox("评估控制")
        control_layout = QHBoxLayout(control_group)
        
        self.btn_new_eval = QPushButton("新建评估")
        self.btn_new_eval.setStyleSheet("background-color: lightblue; padding: 10px;")
        self.btn_new_eval.clicked.connect(self.new_evaluation)
        control_layout.addWidget(self.btn_new_eval)
        
        self.btn_start = QPushButton("开始评估")
        self.btn_start.setStyleSheet("background-color: lightgreen; padding: 10px;")
        self.btn_start.clicked.connect(self.start_evaluation)
        self.btn_start.setEnabled(False)
        control_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("停止评估")
        self.btn_stop.setStyleSheet("background-color: lightcoral; padding: 10px;")
        self.btn_stop.clicked.connect(self.stop_evaluation)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)
        
        self.btn_load_report = QPushButton("加载报告")
        self.btn_load_report.setStyleSheet("background-color: lightyellow; padding: 10px;")
        self.btn_load_report.clicked.connect(self.load_report)
        control_layout.addWidget(self.btn_load_report)
        
        layout.addWidget(control_group)
        
        # 进度区域
        progress_group = QGroupBox("评估进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 分割器：配置信息和报告
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 配置信息
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.addWidget(QLabel("当前配置:"))
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(150)
        config_layout.addWidget(self.config_text)
        splitter.addWidget(config_widget)
        
        # 报告展示
        self.report_widget = EvaluationReportWidget()
        splitter.addWidget(self.report_widget)
        
        splitter.setSizes([150, 400])
        layout.addWidget(splitter)
        
        # 当前配置
        self.current_config = None
    
    def new_evaluation(self):
        """新建评估"""
        dialog = EvaluationConfigDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.current_config = dialog.get_config()
            
            # 显示配置
            self.config_text.setText(json.dumps(self.current_config, indent=2, ensure_ascii=False))
            
            # 启用开始按钮
            self.btn_start.setEnabled(True)
            self.status_label.setText("配置完成，点击开始评估")
            
            ui_logger.info(f"新建评估配置: {self.current_config['name']}")
    
    def start_evaluation(self):
        """开始评估"""
        if not self.current_config:
            QMessageBox.warning(self, "警告", "请先配置评估参数")
            return
        
        # 创建工作线程
        self.evaluation_worker = EvaluationWorker(self.current_config)
        self.evaluation_worker.progress_updated.connect(self.on_progress_updated)
        self.evaluation_worker.episode_completed.connect(self.on_episode_completed)
        self.evaluation_worker.evaluation_finished.connect(self.on_evaluation_finished)
        self.evaluation_worker.error_occurred.connect(self.on_error_occurred)
        
        # 更新UI状态
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_new_eval.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("评估进行中...")
        
        # 启动工作线程
        self.evaluation_worker.start()
        
        ui_logger.info(f"开始评估: {self.current_config['name']}")
    
    def stop_evaluation(self):
        """停止评估"""
        if self.evaluation_worker and self.evaluation_worker.isRunning():
            self.evaluation_worker.stop()
            self.status_label.setText("评估已停止")
        
        # 更新UI状态
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_new_eval.setEnabled(True)
        
        ui_logger.info("评估已停止")
    
    def on_progress_updated(self, current: int, total: int, message: str):
        """进度更新"""
        progress = int((current / max(1, total)) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_episode_completed(self, episode: int, data: dict):
        """回合完成"""
        ui_logger.debug(f"回合 {episode} 完成: {data}")
    
    def on_evaluation_finished(self, result: dict):
        """评估完成"""
        # 更新UI状态
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_new_eval.setEnabled(True)
        self.progress_bar.setValue(100)
        
        if result.get('success'):
            self.status_label.setText("评估完成!")
            self.report_widget.display_report(result)
            
            # 显示成功消息
            evaluation = result.get('evaluation', {})
            metrics = evaluation.get('metrics', {})
            
            QMessageBox.information(
                self,
                "评估完成",
                f"评估成功完成!\n\n"
                f"成功率: {metrics.get('success_rate', 0):.2%}\n"
                f"总奖励: {metrics.get('total_reward', 0):.2f}\n"
                f"报告已生成"
            )
            
            ui_logger.info(f"评估完成: {evaluation.get('name')}")
        else:
            self.status_label.setText(f"评估失败: {result.get('message', 'Unknown error')}")
            QMessageBox.critical(self, "评估失败", result.get('message', 'Unknown error'))
            
            ui_logger.error(f"评估失败: {result.get('message')}")
    
    def on_error_occurred(self, error: str):
        """发生错误"""
        self.status_label.setText(f"错误: {error}")
        QMessageBox.critical(self, "错误", f"评估过程中发生错误:\n{error}")
        
        # 更新UI状态
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_new_eval.setEnabled(True)
        
        ui_logger.error(f"评估错误: {error}")
    
    def load_report(self):
        """加载报告"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载报告", "", "JSON Files (*.json)"
        )
        
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                self.report_widget.display_report(report)
                self.status_label.setText(f"已加载报告: {path}")
                
                ui_logger.info(f"加载报告: {path}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载报告失败:\n{str(e)}")
                ui_logger.error(f"加载报告失败: {e}")
