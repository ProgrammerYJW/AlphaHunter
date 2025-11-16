#!/usr/bin/env python3
"""
灵析个人量化投研平台 - 主应用文件
Flask后端服务，提供完整的API接口
"""

import os
import sys
import json
import queue
import threading
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_from_directory, render_template_string, request, make_response
from flask_cors import CORS

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 全局变量
log_q = queue.Queue()

# 确保必要的目录存在
os.makedirs('src', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ===== 禁止浏览器缓存 =====
@app.after_request
def no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# ===== 静态页统一托管（带时间戳模板）=====
try:
    FEATURES_HTML = open('features.html', encoding='utf-8').read()
except FileNotFoundError:
    FEATURES_HTML = """
    <!DOCTYPE html>
    <html><head><title>功能中心 - 灵析量化</title></head>
    <body>
        <h1>功能中心</h1>
        <p>功能页面加载中...</p>
        <a href="index.html">返回首页</a>
    </body>
    </html>
    """

# ===== 路由定义 =====
@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/index')
@app.route('/index.html')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/features')
@app.route('/features.html')
def features():
    return render_template_string(FEATURES_HTML, t=int(time.time()))

@app.route('/factor')
@app.route('/factor.html')
def factor():
    return send_from_directory('.', 'factor.html')

@app.route('/strategy')
@app.route('/strategy.html')
def strategy():
    return send_from_directory('.', 'strategy.html')

@app.route('/portfolio')
@app.route('/portfolio.html')
def portfolio():
    return send_from_directory('.', 'portfolio.html')

@app.route('/analytics')
@app.route('/analytics.html')
def analytics():
    return send_from_directory('.', 'analytics.html')

@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ===== 因子研究 API =====
@app.route('/api/factor/run', methods=['POST'])
def run_factor_mining():
    def worker():
        try:
            data = request.get_json()
            algorithm = data.get('algorithm', 'gp')
            
            log_q.put(('log', f'开始因子挖掘，算法: {algorithm}'))
            
            # 模拟因子挖掘过程
            if algorithm == 'gp':
                run_gp_mining()
            elif algorithm == 'rl':
                run_rl_mining()
            else:
                run_hybrid_mining()
                
            log_q.put(('done', '因子挖掘完成'))
        except Exception as e:
            log_q.put(('error', str(e)))
    
    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'status': 'started'})

def run_gp_mining():
    """模拟遗传规划因子挖掘"""
    log_q.put(('log', '初始化遗传规划算法...'))
    time.sleep(1)
    
    # 模拟挖掘过程
    for i in range(10):
        progress = {
            'current': i + 1,
            'total': 10,
            'percentage': (i + 1) * 10
        }
        log_q.put(('progress', progress))
        log_q.put(('log', f'第 {i+1} 代进化完成'))
        
        # 模拟发现因子
        if i % 3 == 0:
            factor = generate_random_factor()
            log_q.put(('factor', factor))
            log_q.put(('log', f'发现高质量因子: {factor["expression"]}'))
        
        time.sleep(0.5)

def run_rl_mining():
    """模拟强化学习因子挖掘"""
    log_q.put(('log', '初始化强化学习算法...'))
    time.sleep(1)
    
    # 模拟训练过程
    for episode in range(5):
        log_q.put(('log', f'第 {episode+1} 轮训练开始'))
        
        for step in range(20):
            progress = {
                'current': episode * 20 + step + 1,
                'total': 100,
                'percentage': (episode * 20 + step + 1)
            }
            log_q.put(('progress', progress))
            
            if step % 5 == 0:
                factor = generate_random_factor()
                log_q.put(('factor', factor))
                log_q.put(('log', f'发现强化学习因子: {factor["expression"]}'))
        
        log_q.put(('log', f'第 {episode+1} 轮训练完成'))
        time.sleep(0.8)

def run_hybrid_mining():
    """模拟混合算法因子挖掘"""
    log_q.put(('log', '初始化混合算法...'))
    time.sleep(1)
    
    # 先运行遗传规划
    log_q.put(('log', '运行遗传规划阶段...'))
    for i in range(5):
        progress = {
            'current': i + 1,
            'total': 10,
            'percentage': (i + 1) * 10
        }
        log_q.put(('progress', progress))
        
        if i % 2 == 0:
            factor = generate_random_factor()
            log_q.put(('factor', factor))
            log_q.put(('log', f'GP阶段发现因子: {factor["expression"]}'))
        
        time.sleep(0.5)
    
    # 再运行强化学习
    log_q.put(('log', '运行强化学习优化阶段...'))
    for i in range(5):
        progress = {
            'current': i + 6,
            'total': 10,
            'percentage': (i + 6) * 10
        }
        log_q.put(('progress', progress))
        
        if i % 2 == 1:
            factor = generate_random_factor()
            log_q.put(('factor', factor))
            log_q.put(('log', f'RL阶段发现因子: {factor["expression"]}'))
        
        time.sleep(0.6)

def generate_random_factor():
    """生成随机因子数据"""
    expressions = [
        "\\frac{close - open}{open}",
        "\\frac{volume}{avg_volume} - 1",
        "\\frac{high - low}{close}",
        "\\frac{close}{ma_20} - 1",
        "\\frac{rsi - 50}{50}",
        "\\frac{macd - signal}{signal}",
        "\\frac{bollinger_upper - close}{bollinger_upper - bollinger_lower}",
        "\\frac{close - min_low}{max_high - min_low}",
        "\\frac{volume * close}{avg_volume * avg_close}",
        "\\frac{sma_10 - sma_30}{sma_30}"
    ]
    
    return {
        "expression": random.choice(expressions),
        "ic": random.uniform(0.05, 0.25),
        "icir": random.uniform(0.5, 2.5),
        "score": random.uniform(0.6, 0.95),
        "complexity": random.randint(3, 15)
    }

@app.route('/api/factor/stream')
def stream():
    def gen():
        while True:
            try:
                typ, payload = log_q.get(timeout=0.3)
                yield f"data: {json.dumps({'type':typ,'payload':payload})}\n\n"
                if typ == 'done': break
            except:
                yield f"data: {json.dumps({'type':'ping'})}\n\n"
    return app.response_class(gen(), mimetype='text/event-stream')

@app.route('/api/factor/result')
def result():
    path = 'src/high_quality_factors.json'
    if os.path.exists(path):
        return jsonify(json.load(open(path, encoding='utf-8')))
    else:
        # 返回模拟数据
        mock_results = [
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
        return jsonify(mock_results)

# ===== 策略研究 API =====
@app.route('/api/strategy/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.get_json()
        strategy = data.get('strategy')
        params = data.get('params', {})
        
        # 模拟回测过程
        time.sleep(2)  # 模拟计算时间
        
        # 生成模拟回测结果
        results = generate_backtest_results(strategy, params)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_backtest_results(strategy, params):
    """生成模拟回测结果"""
    # 生成模拟数据
    if strategy == 'momentum':
        base_return = 0.15
    elif strategy == 'mean_reversion':
        base_return = 0.12
    elif strategy == 'cta':
        base_return = 0.10
    else:
        base_return = 0.08
    
    total_return = base_return + random.uniform(-0.05, 0.05)
    annual_return = total_return
    volatility = random.uniform(0.15, 0.25)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = random.uniform(-0.15, -0.05)
    win_rate = random.uniform(0.55, 0.75)
    trade_count = random.randint(50, 200)
    
    # 生成净值曲线
    days = 252
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, days)
    equity_curve = np.cumprod(1 + daily_returns)
    
    return {
        'totalReturn': total_return,
        'annualReturn': annual_return,
        'sharpeRatio': sharpe_ratio,
        'maxDrawdown': max_drawdown,
        'winRate': win_rate,
        'tradeCount': trade_count,
        'volatility': volatility,
        'equityCurve': equity_curve.tolist(),
        'strategy': strategy,
        'parameters': params
    }

# ===== 投资组合分析 API =====
@app.route('/api/portfolio/analyze', methods=['POST'])
def analyze_portfolio():
    try:
        data = request.get_json()
        assets = data.get('assets', [])
        weights = data.get('weights', [])
        
        # 模拟投资组合分析
        results = generate_portfolio_analysis(assets, weights)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_portfolio_analysis(assets, weights):
    """生成投资组合分析结果"""
    if not assets:
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        weights = [0.25, 0.25, 0.25, 0.25]
    
    # 生成模拟数据
    np.random.seed(42)
    returns = np.random.multivariate_normal(
        mean=[0.001, 0.0008, 0.0012, 0.0006],
        cov=np.array([
            [0.0004, 0.0001, 0.0002, 0.0001],
            [0.0001, 0.0003, 0.0001, 0.00005],
            [0.0002, 0.0001, 0.0005, 0.00015],
            [0.0001, 0.00005, 0.00015, 0.0002]
        ]),
        size=252
    )
    
    # 计算组合收益率
    portfolio_returns = np.sum(returns * weights, axis=1)
    portfolio_value = np.cumprod(1 + portfolio_returns)
    
    # 计算指标
    total_return = portfolio_value[-1] - 1
    annual_return = (1 + total_return) ** (252/252) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 最大回撤
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # 计算各资产的贡献
    asset_contributions = []
    for i, asset in enumerate(assets):
        asset_return = np.mean(returns[:, i]) * 252
        asset_volatility = np.std(returns[:, i]) * np.sqrt(252)
        contribution = weights[i] * asset_return
        
        asset_contributions.append({
            'asset': asset,
            'weight': weights[i],
            'return': asset_return,
            'volatility': asset_volatility,
            'contribution': contribution
        })
    
    return {
        'totalReturn': total_return,
        'annualReturn': annual_return,
        'volatility': volatility,
        'sharpeRatio': sharpe_ratio,
        'maxDrawdown': max_drawdown,
        'portfolioValue': portfolio_value.tolist(),
        'assetContributions': asset_contributions,
        'diversificationRatio': calculate_diversification_ratio(returns, weights)
    }

def calculate_diversification_ratio(returns, weights):
    """计算多样化比率"""
    portfolio_vol = np.sqrt(np.dot(np.array(weights).T, np.dot(np.cov(returns.T), weights)))
    weighted_vol = np.sum(weights * np.sqrt(np.diag(np.cov(returns.T))))
    return portfolio_vol / weighted_vol if weighted_vol > 0 else 1

# ===== 数据API =====
@app.route('/api/market/data')
def get_market_data():
    """获取市场数据"""
    try:
        # 生成模拟市场数据
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        data = []
        
        for symbol in symbols:
            base_price = random.uniform(100, 500)
            prices = []
            current_price = base_price
            
            for i in range(30):  # 30天数据
                change = random.uniform(-0.02, 0.02)
                current_price *= (1 + change)
                prices.append({
                    'date': (datetime.now() - timedelta(days=29-i)).strftime('%Y-%m-%d'),
                    'price': round(current_price, 2),
                    'volume': random.randint(1000000, 10000000)
                })
            
            data.append({
                'symbol': symbol,
                'name': f'Company {symbol}',
                'prices': prices
            })
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== 健康检查 =====
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')