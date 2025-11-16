#!/usr/bin/env python3
"""
灵析个人量化投研平台 - 启动脚本
运行完整的量化投研平台，包含因子挖掘、策略研究、投资组合分析等功能
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from flask import Flask, jsonify, send_from_directory, render_template_string, request
from flask_cors import CORS
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 确保必要的目录存在
os.makedirs('src', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('static', exist_ok=True)

# 创建必要的文件
def create_necessary_files():
    """创建必要的配置文件和数据文件"""
    
    # 创建因子结果文件
    if not os.path.exists('src/high_quality_factors.json'):
        mock_factors = [
            {
                "expression": "\\frac{close - open}{open}",
                "ic": 0.1567,
                "icir": 1.2345,
                "score": 0.8234,
                "complexity": 5
            },
            {
                "expression": "\\frac{volume}{avg_volume} - 1",
                "ic": 0.1234,
                "icir": 0.9876,
                "score": 0.7567,
                "complexity": 8
            },
            {
                "expression": "\\frac{high - low}{close}",
                "ic": 0.0891,
                "icir": 0.6543,
                "score": 0.6789,
                "complexity": 4
            }
        ]
        with open('src/high_quality_factors.json', 'w', encoding='utf-8') as f:
            json.dump(mock_factors, f, ensure_ascii=False, indent=2)
    
    # 创建配置文件
    if not os.path.exists('config.py'):
        config_content = '''"""配置文件"""

class Config:
    # 数据配置
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    
    # 遗传规划配置
    GP_POPULATION_SIZE = 100
    GP_GENERATIONS = 50
    GP_CROSSOVER_RATE = 0.7
    GP_MUTATION_RATE = 0.3
    
    # 强化学习配置
    RL_EPISODES = 1000
    RL_LEARNING_RATE = 0.001
    RL_GAMMA = 0.99
    
    # 因子评估配置
    IC_THRESHOLD = 0.05
    ICIR_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.6
    
    # 策略配置
    BACKTEST_START_DATE = "2020-01-01"
    BACKTEST_END_DATE = "2023-12-31"
    INITIAL_CAPITAL = 1000000
'''
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)

# 创建应用实例
def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # 静态文件路由
    @app.route('/')
    def root():
        return send_from_directory('.', 'index.html')
    
    @app.route('/<path:filename>')
    def static_files(filename):
        if filename.endswith('.html') or filename.endswith('.css'):
            return send_from_directory('.', filename)
        return send_from_directory('static', filename)
    
    # API路由
    @app.route('/api/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    return app

def main():
    """主函数"""
    print("=" * 60)
    print("灵析个人量化投研平台")
    print("=" * 60)
    
    # 创建必要文件
    create_necessary_files()
    
    # 检查Python依赖
    print("检查依赖...")
    required_packages = ['flask', 'flask_cors', 'numpy', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return
    
    print("依赖检查完成 ✓")
    
    # 启动Flask应用
    print("启动Web服务器...")
    app = create_app()
    
    # 在新线程中启动服务器
    def run_server():
        app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(2)
    
    print("Web服务器启动完成 ✓")
    print("应用运行在: http://localhost:5000")
    
    # 自动打开浏览器
    try:
        webbrowser.open('http://localhost:5000')
        print("已自动打开浏览器")
    except:
        print("请手动打开浏览器访问: http://localhost:5000")
    
    print("\n按 Ctrl+C 停止服务器")
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
        print("感谢使用灵析量化投研平台！")

if __name__ == '__main__':
    main()