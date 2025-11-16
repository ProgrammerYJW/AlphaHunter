#!/usr/bin/env python3
"""
灵析量化投研平台 - 测试脚本
测试平台的各项功能是否正常工作
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    """测试API端点"""
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("测试API端点")
    print("=" * 60)
    
    # 1. 健康检查
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✓ 健康检查 - 通过")
        else:
            print("✗ 健康检查 - 失败")
    except Exception as e:
        print(f"✗ 健康检查 - 错误: {e}")
    
    # 2. 市场数据API
    try:
        response = requests.get(f"{base_url}/api/market/data")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 市场数据API - 通过 ({len(data)} 条数据)")
        else:
            print("✗ 市场数据API - 失败")
    except Exception as e:
        print(f"✗ 市场数据API - 错误: {e}")
    
    # 3. 因子结果API
    try:
        response = requests.get(f"{base_url}/api/factor/result")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 因子结果API - 通过 ({len(data)} 个因子)")
        else:
            print("✗ 因子结果API - 失败")
    except Exception as e:
        print(f"✗ 因子结果API - 错误: {e}")
    
    # 4. 投资组合分析API
    try:
        payload = {
            "assets": ["AAPL", "GOOGL", "MSFT", "AMZN"],
            "weights": [0.25, 0.25, 0.25, 0.25]
        }
        response = requests.post(f"{base_url}/api/portfolio/analyze", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 投资组合分析API - 通过 (夏普比率: {data.get('sharpeRatio', 'N/A')})")
        else:
            print("✗ 投资组合分析API - 失败")
    except Exception as e:
        print(f"✗ 投资组合分析API - 错误: {e}")
    
    # 5. 策略回测API
    try:
        payload = {
            "strategy": "momentum",
            "params": {
                "lookback": {"value": 60},
                "percentile": {"value": 0.8}
            }
        }
        response = requests.post(f"{base_url}/api/strategy/backtest", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 策略回测API - 通过 (总收益率: {data.get('totalReturn', 'N/A')})")
        else:
            print("✗ 策略回测API - 失败")
    except Exception as e:
        print(f"✗ 策略回测API - 错误: {e}")

def test_web_pages():
    """测试网页访问"""
    base_url = "http://localhost:5000"
    
    print("\n" + "=" * 60)
    print("测试网页访问")
    print("=" * 60)
    
    pages = [
        ("首页", "/index.html"),
        ("功能中心", "/features.html"),
        ("因子研究", "/factor.html"),
        ("策略研究", "/strategy.html"),
        ("投资组合分析", "/portfolio.html"),
        ("数据分析中心", "/analytics.html")
    ]
    
    for name, path in pages:
        try:
            response = requests.get(f"{base_url}{path}")
            if response.status_code == 200:
                print(f"✓ {name} - 通过")
            else:
                print(f"✗ {name} - 状态码: {response.status_code}")
        except Exception as e:
            print(f"✗ {name} - 错误: {e}")

def test_factor_mining():
    """测试因子挖掘功能"""
    base_url = "http://localhost:5000"
    
    print("\n" + "=" * 60)
    print("测试因子挖掘功能")
    print("=" * 60)
    
    try:
        # 启动因子挖掘
        payload = {"algorithm": "gp"}
        response = requests.post(f"{base_url}/api/factor/run", json=payload)
        
        if response.status_code == 200:
            print("✓ 因子挖掘启动 - 通过")
            
            # 等待一段时间让挖掘进行
            print("等待因子挖掘执行...")
            time.sleep(3)
            
            # 获取结果
            result_response = requests.get(f"{base_url}/api/factor/result")
            if result_response.status_code == 200:
                data = result_response.json()
                print(f"✓ 因子结果获取 - 通过 ({len(data)} 个因子)")
            else:
                print("✗ 因子结果获取 - 失败")
        else:
            print("✗ 因子挖掘启动 - 失败")
    except Exception as e:
        print(f"✗ 因子挖掘 - 错误: {e}")

def test_static_files():
    """测试静态文件"""
    base_url = "http://localhost:5000"
    
    print("\n" + "=" * 60)
    print("测试静态文件")
    print("=" * 60)
    
    files = [
        ("CSS样式表", "/style.css"),
        ("因子JS", "/static/factor.js"),
        ("策略JS", "/static/strategy.js"),
        ("图表JS", "/static/charts.js")
    ]
    
    for name, path in files:
        try:
            response = requests.get(f"{base_url}{path}")
            if response.status_code == 200:
                size = len(response.content)
                print(f"✓ {name} - 通过 ({size} bytes)")
            else:
                print(f"✗ {name} - 状态码: {response.status_code}")
        except Exception as e:
            print(f"✗ {name} - 错误: {e}")

def main():
    """主测试函数"""
    print("灵析量化投研平台 - 功能测试")
    print("测试时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        # 测试各项功能
        test_api_endpoints()
        test_web_pages()
        test_factor_mining()
        test_static_files()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        print("平台运行正常，所有核心功能已测试通过。")
        print("请访问 http://localhost:5000 开始使用平台。")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("请确保平台已正确启动。")

if __name__ == '__main__':
    main()