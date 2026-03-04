"""
日志配置模块
统一管理整个项目的日志系统
"""
import logging
import os
from datetime import datetime
from pathlib import Path


class ProjectLogger:
    """项目日志管理器"""
    
    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        """
        初始化日志管理器
        
        参数:
            name: 日志记录器名称
            log_dir: 日志目录
            level: 日志级别
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 避免重复添加处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（按日期分割）
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)


# 创建全局日志记录器
def get_logger(name: str, log_dir: str = "logs") -> ProjectLogger:
    """
    获取或创建日志记录器
    
    参数:
        name: 日志记录器名称
        log_dir: 日志目录
        
    返回:
        ProjectLogger实例
    """
    return ProjectLogger(name, log_dir)


# 预定义的日志记录器
system_logger = get_logger('system')
training_logger = get_logger('training')
inference_logger = get_logger('inference')
ui_logger = get_logger('ui')
backend_logger = get_logger('backend')
environment_logger = get_logger('environment')
agent_logger = get_logger('agent')