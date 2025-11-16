import numpy as np
import pandas as pd
from scipy import stats
import config


class FactorEvaluator:
    """因子评估器"""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def evaluate_alpha_metrics(self, factor_values, returns):
        """评估alpha指标"""
        metrics = {}

        valid_mask = ~np.isnan(factor_values) & ~np.isnan(returns)
        valid_count = np.sum(valid_mask)

        if valid_count > 10:
            factor_valid = factor_values[valid_mask]
            returns_valid = returns[valid_mask]

            # 信息系数
            try:
                ic_value, _ = stats.spearmanr(factor_valid, returns_valid)
                metrics['IC'] = ic_value if not np.isnan(ic_value) else 0
            except:
                metrics['IC'] = 0

            # ICIR
            metrics['ICIR'] = abs(metrics['IC']) / 0.1 if metrics['IC'] != 0 else 0

            # 多空收益
            try:
                if len(np.unique(factor_valid)) > 5:  # 确保有足够的分位数
                    deciles = pd.qcut(factor_valid, 10, labels=False, duplicates='drop')
                    if len(np.unique(deciles)) > 1:
                        top_return = np.mean(returns_valid[deciles == deciles.max()])
                        bottom_return = np.mean(returns_valid[deciles == deciles.min()])
                        metrics['long_short_return'] = top_return - bottom_return
                    else:
                        metrics['long_short_return'] = 0
                else:
                    metrics['long_short_return'] = 0
            except:
                metrics['long_short_return'] = 0
        else:
            metrics['IC'] = 0
            metrics['ICIR'] = 0
            metrics['long_short_return'] = 0

        return metrics

    def evaluate_risk_metrics(self, factor_returns):
        """评估风险指标"""
        metrics = {}

        if len(factor_returns) > 10 and not np.all(np.isnan(factor_returns)):
            factor_returns_clean = factor_returns[~np.isnan(factor_returns)]

            if len(factor_returns_clean) > 0:
                metrics['volatility'] = np.std(factor_returns_clean)

                cumulative = np.cumprod(1 + factor_returns_clean)
                peak = np.maximum.accumulate(cumulative)
                drawdown = (peak - cumulative) / peak
                metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0

                metrics['sharpe_ratio'] = (
                        np.mean(factor_returns_clean) / (np.std(factor_returns_clean) + 1e-8)
                )
            else:
                metrics['volatility'] = 0
                metrics['max_drawdown'] = 0
                metrics['sharpe_ratio'] = 0
        else:
            metrics['volatility'] = 0
            metrics['max_drawdown'] = 0
            metrics['sharpe_ratio'] = 0

        return metrics

    def comprehensive_evaluation(self, factors_with_data):
        """综合评估因子"""
        high_quality_factors = []

        for factor_info in factors_with_data:
            alpha_metrics = self.evaluate_alpha_metrics(
                factor_info['values'],
                factor_info['returns']
            )

            risk_metrics = self.evaluate_risk_metrics(factor_info.get('factor_returns', []))

            # 综合评分
            alpha_score = alpha_metrics['ICIR'] + alpha_metrics['long_short_return'] * 10
            risk_score = -risk_metrics['max_drawdown'] * 10 + risk_metrics['sharpe_ratio']
            composite_score = alpha_score + risk_score

            # 筛选高质量因子
            if (abs(alpha_metrics['IC']) >= config.Config.MIN_IC and
                    alpha_metrics['ICIR'] >= config.Config.MIN_ICIR and
                    abs(risk_metrics['max_drawdown']) <= config.Config.MAX_DRAWDOWN):
                high_quality_factors.append({
                    'expression': factor_info['expression'],
                    'IC': alpha_metrics['IC'],
                    'ICIR': alpha_metrics['ICIR'],
                    'long_short_return': alpha_metrics['long_short_return'],
                    'max_drawdown': risk_metrics['max_drawdown'],
                    'sharpe_ratio': risk_metrics['sharpe_ratio'],
                    'composite_score': composite_score,
                    'method': factor_info['method']
                })

        high_quality_factors.sort(key=lambda x: x['composite_score'], reverse=True)
        return high_quality_factors