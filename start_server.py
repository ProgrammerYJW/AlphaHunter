#!/usr/bin/env python3
"""
灵析量化投研平台 - 简化启动脚本
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser

# 切换到当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        sys.exit(1)
    print(f"Python版本: {sys.version}")

def install_requirements():
    """安装依赖包"""
    print("检查依赖包...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("依赖包安装完成 ✓")
    except Exception as e:
        print(f"依赖包安装失败: {e}")
        print("请手动运行: pip install -r requirements.txt")

def generate_demo_data():
    """生成演示数据"""
    print("生成演示数据...")
    try:
        subprocess.check_call([sys.executable, 'generate_demo_data.py'])
        print("演示数据生成完成 ✓")
    except Exception as e:
        print(f"演示数据生成失败: {e}")

def start_flask_app():
    """启动Flask应用"""
    print("启动Flask应用...")
    try:
        # 使用subprocess启动Flask应用
        env = os.environ.copy()
        env['FLASK_ENV'] = 'development'
        
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], env=env)
        
        return process
    except Exception as e:
        print(f"启动Flask应用失败: {e}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("灵析个人量化投研平台")
    print("=" * 60)
    
    # 1. 检查Python版本
    check_python_version()
    
    # 2. 安装依赖
    install_requirements()
    
    # 3. 生成演示数据
    generate_demo_data()
    
    # 4. 启动Flask应用
    flask_process = start_flask_app()
    
    if flask_process:
        print("Flask应用启动完成 ✓")
        print("应用运行在: http://localhost:5000")
        
        # 等待服务器启动
        time.sleep(3)
        
        # 自动打开浏览器
        try:
            webbrowser.open('http://localhost:5000')
            print("已自动打开浏览器")
        except:
            print("请手动打开浏览器访问: http://localhost:5000")
        
        print("\n按 Ctrl+C 停止服务器")
        
        try:
            # 等待进程
            flask_process.wait()
        except KeyboardInterrupt:
            print("\n正在关闭服务器...")
            flask_process.terminate()
            flask_process.wait()
            print("服务器已关闭")
    else:
        print("启动失败，请检查错误信息")

if __name__ == '__main__':
    main()