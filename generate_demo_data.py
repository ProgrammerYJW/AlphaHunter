#!/usr/bin/env python3
"""
演示数据生成脚本
为灵析量化投研平台生成示例数据
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta

def generate_market_data():
    """生成市场数据"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    names = ['苹果', '谷歌', '微软', '亚马逊', '特斯拉', '英伟达', 'Meta', '奈飞']
    
    market_data = []
    
    for i, symbol in enumerate(symbols):
        base_price = random.uniform(100, 500)
        prices = []
        current_price = base_price
        
        # 生成30天数据
        for day in range(30):
            date = datetime.now() - timedelta(days=29-day)
            
            # 模拟价格波动
            daily_return = random.normalvariate(0.001, 0.02)
            current_price *= (1 + daily_return)
            
            prices.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(current_price, 2),
                'volume': random.randint(1000000, 10000000),
                'open': round(current_price * (1 + random.uniform(-0.01, 0.01)), 2),
                'high': round(current_price * (1 + random.uniform(0, 0.02)), 2),
                'low': round(current_price * (1 + random.uniform(-0.02, 0)), 2),
                'close': round(current_price, 2)
            })
        
        market_data.append({
            'symbol': symbol,
            'name': names[i],
            'prices': prices
        })
    
    return market_data

def generate_factor_expressions():
    """生成因子表达式"""
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
        "\\frac{sma_10 - sma_30}{sma_30}",
        "\\frac{close - sma_20}{sma_20}",
        "\\frac{volume - ma_volume_20}{ma_volume_20}",
        "\\frac{high - close}{high - low}",
        "\\frac{close - low}{high - low}",
        "\\frac{open - low}{high - low}"
    ]
    return expressions

def generate_factor_results():
    """生成因子结果数据"""
    expressions = generate_factor_expressions()
    factors = []
    
    for i, expr in enumerate(expressions[:10]):
        factor = {
            "expression": expr,
            "ic": random.uniform(0.05, 0.25),
            "icir": random.uniform(0.5, 2.5),
            "score": random.uniform(0.6, 0.95),
            "complexity": random.randint(3, 15)
        }
        factors.append(factor)
    
    return factors

def generate_portfolio_performance():
    """生成组合表现数据"""
    # 生成252个交易日的数据
    days = 252
    np.random.seed(42)
    
    # 生成随机收益率
    daily_returns = np.random.normal(0.001, 0.02, days)
    
    # 计算净值曲线
    equity_curve = np.cumprod(1 + daily_returns)
    
    # 计算指标
    total_return = equity_curve[-1] - 1
    annual_return = (1 + total_return) ** (252/days) - 1
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 最大回撤
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # 胜率
    positive_days = np.sum(daily_returns > 0)
    win_rate = positive_days / len(daily_returns)
    
    # 资产配置
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    weights = [0.25, 0.25, 0.25, 0.25]
    
    asset_contributions = []
    for i, asset in enumerate(assets):
        asset_return = np.mean(daily_returns) * 252
        asset_volatility = np.std(daily_returns) * np.sqrt(252)
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
        'winRate': win_rate,
        'portfolioValue': equity_curve.tolist(),
        'assetContributions': asset_contributions,
        'diversificationRatio': 0.85
    }

def generate_strategy_backtests():
    """生成策略回测结果"""
    strategies = ['topk_drop', 'momentum', 'mean_reversion', 'cta']
    results = {}
    
    for strategy in strategies:
        # 生成不同的收益率
        if strategy == 'momentum':
            base_return = 0.15
        elif strategy == 'mean_reversion':
            base_return = 0.12
        elif strategy == 'cta':
            base_return = 0.10
        else:
            base_return = 0.08
        
        # 添加随机波动
        total_return = base_return + random.uniform(-0.05, 0.05)
        annual_return = total_return
        volatility = random.uniform(0.15, 0.25)
        sharpe_ratio = annual_return / volatility
        max_drawdown = random.uniform(-0.15, -0.05)
        win_rate = random.uniform(0.55, 0.75)
        trade_count = random.randint(50, 200)
        
        results[strategy] = {
            'totalReturn': total_return,
            'annualReturn': annual_return,
            'sharpeRatio': sharpe_ratio,
            'maxDrawdown': max_drawdown,
            'winRate': win_rate,
            'tradeCount': trade_count,
            'volatility': volatility
        }
    
    return results

def main():
    """主函数 - 生成所有演示数据"""
    print("正在生成演示数据...")
    
    # 1. 生成市场数据
    print("1. 生成市场数据...")
    market_data = generate_market_data()
    with open('data/market_data.json', 'w', encoding='utf-8') as f:
        json.dump(market_data, f, ensure_ascii=False, indent=2)
    
    # 2. 生成因子结果
    print("2. 生成因子结果...")
    factor_results = generate_factor_results()
    with open('src/high_quality_factors.json', 'w', encoding='utf-8') as f:
        json.dump(factor_results, f, ensure_ascii=False, indent=2)
    
    # 3. 生成组合表现数据
    print("3. 生成组合表现数据...")
    portfolio_data = generate_portfolio_performance()
    with open('data/portfolio_performance.json', 'w', encoding='utf-8') as f:
        json.dump(portfolio_data, f, ensure_ascii=False, indent=2)
    
    # 4. 生成策略回测结果
    print("4. 生成策略回测结果...")
    backtest_results = generate_strategy_backtests()
    with open('data/backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(backtest_results, f, ensure_ascii=False, indent=2)
    
    print("演示数据生成完成！")
    print(f"- 市场数据: {len(market_data)} 只股票")
    print(f"- 因子结果: {len(factor_results)} 个因子")
    print(f"- 组合数据: 已生成")
    print(f"- 回测结果: {len(backtest_results)} 个策略")

if __name__ == '__main__':
    main()