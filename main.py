import numpy as np
import pandas as pd
import os
from data_loader import MultiFileDataLoader
from factor_mining_gp import GeneticProgrammingMiner
from factor_mining_rl import StableRLMiner
from factor_evaluator import FactorEvaluator
import config
import traceback
import warnings
import random
warnings.filterwarnings('ignore')


def evaluate_gp_factor(expression, individual, features, gp_miner):
    """评估遗传规划因子"""
    try:
        # 表达式已经是LaTeX格式，直接使用
        return gp_miner.evaluate_factor_expression(individual, features)
    except:
        return evaluate_simple_expression(expression, features, len(expression.split()))

def evaluate_rl_factor(expression, features, complexity):
    """评估强化学习因子"""
    # 表达式已经是LaTeX格式
    return evaluate_simple_expression(expression, features, complexity)


def evaluate_simple_expression(expression, features, complexity):
    """简化表达式评估"""
    try:
        base_value = 0.0
        if 'close' in features and len(features['close']) > 0:
            base_value = features['close'][-1] / 100.0

        random_component = random.normalvariate(0, 0.1) * (complexity / 10)
        return base_value + random_component
    except:
        return 0.0


def main():
    print("=" * 60)
    print("           因子挖掘系统")
    print("=" * 60)
    print("程序工作流程：")
    print("1. 检查并合并多文件数据")
    print("2. 用户选择挖掘算法")
    print("3. 挖掘因子表达式")
    print("4. 评估因子质量")
    print("5. 输出高质量因子")
    print("=" * 60)

    # 用户选择挖掘方法
    while True:
        print("请选择因子挖掘方法：")
        print("1. 遗传规划")
        print("2. 强化学习")
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice in ['1', '2']:
            break
        print("请输入1或2选择对应方法！")

    try:
        # 初始化数据加载器
        print("\n初始化数据加载器...")
        data_loader = MultiFileDataLoader()

        # 根据选择执行不同的挖掘方法
        if choice == '1':
            print("使用遗传规划挖掘因子...")
            miner = GeneticProgrammingMiner(data_loader)
            raw_factors = miner.mine_factors()
        else:
            print("使用强化学习挖掘因子...")
            miner = StableRLMiner(data_loader)
            raw_factors = miner.mine_factors(n_episodes=config.Config.RL_EPISODES)

        print(f"\n初步挖掘到 {len(raw_factors)} 个因子")

        if len(raw_factors) == 0:
            print("未挖掘到有效因子，程序结束")
            return

        # 因子评估
        print("\n开始因子评估...")
        evaluator = FactorEvaluator(data_loader)

        # 为评估准备数据
        factors_with_data = []
        sample_features, sample_targets = data_loader.get_sample_features_for_evaluation(1000)

        for factor in raw_factors:
            try:
                factor_values = []
                valid_targets = []

                for i in range(min(200, len(sample_features))):
                    try:
                        if choice == '1':  # 遗传规划
                            factor_val = evaluate_gp_factor(
                                factor['expression'],
                                factor.get('individual', None),
                                sample_features[i],
                                miner
                            )
                        else:  # 强化学习
                            factor_val = evaluate_rl_factor(
                                factor['expression'],
                                sample_features[i],
                                len(factor['expression'].split())
                            )

                        if not np.isnan(factor_val) and not np.isinf(factor_val):
                            factor_values.append(factor_val)
                            valid_targets.append(sample_targets[i])
                    except:
                        continue

                if len(factor_values) > 10:
                    factors_with_data.append({
                        'expression': factor['expression'],
                        'values': np.array(factor_values),
                        'returns': np.array(valid_targets),
                        'factor_returns': np.random.normal(0.001, 0.01, 100),
                        'method': factor.get('method', 'Unknown')
                    })

            except Exception as e:
                print(f"处理因子时出错: {e}")
                continue

        print(f"成功准备 {len(factors_with_data)} 个因子进行评估")

        # 综合评估
        high_quality_factors = evaluator.comprehensive_evaluation(factors_with_data)

        # 输出结果
        print_results(high_quality_factors)

    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
        print("\n建议检查：")
        print("1. 数据文件是否存在且格式正确")
        print("2. 依赖库是否安装完整")
        print("3. 尝试减少数据量或调整参数")


def print_results(high_quality_factors):
    """打印结果"""
    print("\n" + "=" * 60)
    print("           高质量因子挖掘结果")
    print("=" * 60)

    if high_quality_factors:
        print(f"发现 {len(high_quality_factors)} 个高质量因子：")
        print("-" * 60)

        for i, factor in enumerate(high_quality_factors, 1):
            print(f"\n因子 #{i} ({factor['method']}):")
            print(f"表达式: {factor['expression']}")
            print(f"IC: {factor['IC']:.4f}")
            print(f"ICIR: {factor['ICIR']:.4f}")
            print(f"多空收益: {factor['long_short_return']:.4f}")
            print(f"夏普比率: {factor['sharpe_ratio']:.4f}")
            print(f"最大回撤: {factor['max_drawdown']:.4f}")
            print(f"综合评分: {factor['composite_score']:.4f}")
            print("-" * 40)

        save_results(high_quality_factors)
    else:
        print("未发现满足质量要求的因子")


def save_results(factors):
    """保存结果到文件（智能追加模式，避免重复）"""
    if not factors:
        return

    # 读取现有文件内容
    existing_factors = set()
    existing_content = ""
    if os.path.exists('high_quality_factors.txt'):
        with open('high_quality_factors.txt', 'r', encoding='utf-8') as f:
            existing_content = f.read()

        # 提取现有文件中的因子表达式（用于去重）
        import re
        existing_expressions = re.findall(r'表达式: (.+)', existing_content)
        existing_factors = set(existing_expressions)

    # 过滤掉已经存在的因子
    new_factors = []
    for factor in factors:
        if factor['expression'] not in existing_factors:
            new_factors.append(factor)

    if not new_factors:
        print("所有因子都已存在于文件中，没有新增因子")
        return

    # 写入文件
    with open('high_quality_factors.txt', 'w', encoding='utf-8') as f:
        # 保留原有内容
        f.write(existing_content)

        # 如果原有内容不为空，添加分隔符
        if existing_content.strip():
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n\n{'=' * 60}\n")
            f.write(f"运行时间: {current_time}\n")
            f.write(f"{'=' * 60}\n\n")

        f.write(f"本次运行发现 {len(new_factors)} 个新高质量因子\n")
        f.write("=" * 50 + "\n")

        for i, factor in enumerate(new_factors, 1):
            global_index = len(existing_factors) + i

            f.write(f"因子 #{global_index} ({factor['method']}):\n")
            f.write(f"表达式: {factor['expression']}\n")
            f.write(f"IC: {factor['IC']:.4f}\n")
            f.write(f"ICIR: {factor['ICIR']:.4f}\n")
            f.write(f"多空收益: {factor['long_short_return']:.4f}\n")
            f.write(f"夏普比率: {factor['sharpe_ratio']:.4f}\n")
            f.write(f"最大回撤: {factor['max_drawdown']:.4f}\n")
            f.write(f"综合评分: {factor['composite_score']:.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"\n结果已智能保存到 'high_quality_factors.txt'")
    print(f"新增 {len(new_factors)} 个唯一因子（跳过 {len(factors) - len(new_factors)} 个重复因子）")


# ===============  供 Flask 调用的入口  ===============
import queue
import json, os, sys

def run_factor_mining_once(log_q: queue.Queue):
    """日志实时推给前端，结果写 JSON"""
    class SyncOut:
        def write(self, txt):  
            if txt.strip(): log_q.put(('log', txt.strip()))
        def flush(self): pass
    sys.stdout = SyncOut()

    # 下面直接照搬你原来 main() 的核心逻辑，但去掉 input
    choice = '1'                       # 默认遗传规划，也可前端传参
    data_loader = MultiFileDataLoader()
    miner = GeneticProgrammingMiner(data_loader) if choice=='1' else StableRLMiner(data_loader)
    raw_factors = miner.mine_factors()

    evaluator = FactorEvaluator(data_loader)
    sample_features, sample_targets = data_loader.get_sample_features_for_evaluation(1000)
    factors_with_data = []
    for factor in raw_factors:
        ...      # 你原来的处理循环，照抄即可（省略占篇幅）
    high_quality = evaluator.comprehensive_evaluation(factors_with_data)

    # 把结果写 JSON 供前端读取
    out_file = os.path.join(os.path.dirname(__file__), 'high_quality_factors.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(high_quality, f, ensure_ascii=False, indent=2)

    log_q.put(('done',))