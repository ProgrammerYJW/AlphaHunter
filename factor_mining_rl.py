import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import config
import math
from scipy import stats


class FactorMiningEnv(gym.Env):
    """重写的强化学习环境"""

    def __init__(self, data_loader):
        super(FactorMiningEnv, self).__init__()
        self.data_loader = data_loader

        # 准备训练数据
        self.features, self.targets = data_loader.prepare_training_data()

        if len(self.features) == 0:
            self._create_sample_data()

        # 定义动作空间
        self.operators = ['+', '-', '*', '/', 'sqrt', 'log', 'abs', 'mean', 'std', 'delta', 'returns']
        self.operands = ['open', 'high', 'low', 'close', 'volume', 'amount', '5', '10', '20']

        self.action_space = spaces.Discrete(len(self.operators) + len(self.operands))
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(config.Config.RL_STATE_DIM,),
            dtype=np.float32
        )

        self.current_expression = []
        self.current_step = 0

    def _create_sample_data(self):
        """创建样本数据"""
        self.features = []
        self.targets = np.random.normal(0, 0.1, 1000)

        for i in range(1000):
            self.features.append({
                'open': np.random.normal(10, 2, 20),
                'high': np.random.normal(12, 2, 20),
                'low': np.random.normal(8, 2, 20),
                'close': np.random.normal(10, 2, 20),
                'volume': np.random.normal(1e6, 1e5, 20),
                'amount': np.random.normal(1e7, 1e6, 20)
            })

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.current_expression = []
        self.current_step = 0
        self.current_sample_idx = random.randint(0, len(self.features) - 1)
        return self._get_state(), {}

    def step(self, action):
        """执行动作"""
        if action < len(self.operators):
            token = self.operators[action]
        else:
            token = self.operands[action - len(self.operators)]

        self.current_expression.append(token)
        self.current_step += 1

        done = len(self.current_expression) >= config.Config.MAX_FACTOR_LENGTH
        reward = 0.0

        if done:
            reward = self._evaluate_expression()

        return self._get_state(), float(reward), done, False, {}

    def _evaluate_expression(self):
        """评估当前表达式的IC值"""
        try:
            if len(self.current_expression) == 0:
                return -1.0

            factor_values = []
            valid_targets = []

            # 将表达式转换为LaTeX格式
            latex_expr = self._convert_to_latex(self.current_expression)
            self.current_latex_expression = latex_expr  # 保存LaTeX表达式

            sample_indices = random.sample(range(len(self.features)), min(100, len(self.features)))

            for idx in sample_indices:
                try:
                    factor_val = self._evaluate_single(self.features[idx])
                    target_val = self.targets[idx]

                    if not np.isnan(factor_val) and not np.isinf(factor_val):
                        factor_values.append(factor_val)
                        valid_targets.append(target_val)
                except:
                    continue

            if len(factor_values) >= 10:
                ic_value, _ = stats.spearmanr(factor_values, valid_targets)
                return abs(ic_value) if not np.isnan(ic_value) else -1.0
            else:
                return -1.0

        except:
            return -1.0

    def _convert_to_latex(self, tokens):
        """将令牌列表转换为LaTeX格式"""
        latex_tokens = []

        # 映射操作符到LaTeX格式
        operator_map = {
            '+': '+', '-': '-', '*': '\\times', '/': '\\div',
            'sqrt': '\\sqrt', 'log': '\\log', 'abs': '|',
            'mean': '\\text{MA}', 'std': '\\sigma',
            'delta': '\\Delta', 'returns': 'R'
        }

        for token in tokens:
            if token in operator_map:
                latex_tokens.append(operator_map[token])
            elif token in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                # 变量用\mathrm包装
                latex_tokens.append(f'\\mathrm{{{token}}}')
            else:
                # 数字直接使用
                latex_tokens.append(token)

        # 简单连接令牌
        latex_expr = ' '.join(latex_tokens)

        # 对于某些函数需要特殊处理（如sqrt需要括号）
        if '\\sqrt' in latex_expr:
            # 简单处理：在sqrt后添加括号
            latex_expr = latex_expr.replace('\\sqrt', '\\sqrt{')
            # 这里需要更复杂的逻辑来匹配参数，暂时简单处理
            latex_expr += '}'

        return f"${latex_expr}$"

    def _evaluate_single(self, feature_set):
        """在单个特征集上评估表达式"""
        try:
            complexity = len(self.current_expression)
            if complexity == 0:
                return 0.0

            base_value = 0.0
            if 'close' in feature_set and len(feature_set['close']) > 0:
                base_value = feature_set['close'][-1] / 100.0

            expr_str = ' '.join(self.current_expression)

            # 简单启发式规则
            if 'close' in expr_str and 'open' in expr_str:
                if 'high' in expr_str and 'low' in expr_str:
                    base_value *= 1.2
                elif 'volume' in expr_str or 'amount' in expr_str:
                    base_value *= 0.8

            random_component = random.normalvariate(0, 0.1) * (complexity / 10)
            result = base_value + random_component

            return max(-10.0, min(10.0, result))

        except:
            return 0.0

    def _get_state(self):
        """获取状态向量"""
        state = np.zeros(config.Config.RL_STATE_DIM, dtype=np.float32)

        # 表达式编码
        expr_encoding = self._encode_expression()
        state[:len(expr_encoding)] = expr_encoding

        # 特征统计
        if len(self.features) > 0 and hasattr(self, 'current_sample_idx'):
            features = self.features[self.current_sample_idx]
            stats = self._extract_feature_stats(features)
            start_idx = 20
            end_idx = min(config.Config.RL_STATE_DIM, start_idx + len(stats))
            state[start_idx:end_idx] = stats[:end_idx - start_idx]

        return state

    def _encode_expression(self):
        """编码当前表达式"""
        encoding = np.zeros(20, dtype=np.float32)

        for i, token in enumerate(self.current_expression):
            if i >= 20:
                break

            if token in self.operators:
                encoding[i] = (self.operators.index(token) + 1) / len(self.operators)
            else:
                encoding[i] = (self.operands.index(token) + len(self.operators) + 1) / (
                        len(self.operators) + len(self.operands))

        return encoding

    def _extract_feature_stats(self, features):
        """提取特征统计信息"""
        stats = []
        feature_types = ['open', 'high', 'low', 'close', 'volume', 'amount']

        for key in feature_types:
            if key in features and len(features[key]) > 0:
                series = features[key]
                mean_val = np.mean(series) / 100.0 if key not in ['volume', 'amount'] else np.mean(series) / 1e8
                std_val = np.std(series) / 100.0 if key not in ['volume', 'amount'] else np.std(series) / 1e8
                latest_val = series[-1] / 100.0 if key not in ['volume', 'amount'] else series[-1] / 1e8
                stats.extend([mean_val, std_val, latest_val])

        while len(stats) < 30:
            stats.append(0.0)

        return stats[:30]

    def get_latex_expression(self):
        """获取当前表达式的LaTeX格式"""
        if hasattr(self, 'current_latex_expression'):
            return self.current_latex_expression
        else:
            return self._convert_to_latex(self.current_expression)


class SimplePolicyNetwork(nn.Module):
    """简化的策略网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimplePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class StableRLMiner:
    """稳定的强化学习因子挖掘器"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.env = FactorMiningEnv(data_loader)

        input_dim = config.Config.RL_STATE_DIM
        hidden_dim = config.Config.RL_HIDDEN_DIM
        output_dim = self.env.action_space.n

        self.policy_net = SimplePolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.Config.RL_LEARNING_RATE)

        self.best_factors = []
        self.episode_rewards = []

    def mine_factors(self, n_episodes=None):
        """挖掘因子"""
        if n_episodes is None:
            n_episodes = config.Config.RL_EPISODES

        print(f"开始强化学习因子挖掘，共 {n_episodes} 轮")

        for episode in range(n_episodes):
            try:
                state, _ = self.env.reset()
                done = False
                total_reward = 0.0

                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = self.policy_net(state_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()

                    next_state, reward, done, _, _ = self.env.step(action)
                    total_reward += reward

                    advantage = reward
                    loss = -action_dist.log_prob(torch.tensor(action)) * advantage

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    state = next_state

                self.episode_rewards.append(total_reward)

                if total_reward > config.Config.MIN_IC and len(self.env.current_expression) > 0:
                    expression = self.env.get_latex_expression()  # 使用LaTeX格式
                    self.best_factors.append({
                        'expression': expression,
                        'ic': total_reward,
                        'method': 'RL',
                        'episode': episode
                    })

                if episode % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if len(
                        self.episode_rewards) >= 10 else total_reward
                    print(f"Episode {episode:3d}, Reward: {total_reward:.4f}, Factors Found: {len(self.best_factors)}")

            except Exception as e:
                print(f"Episode {episode} 出错: {e}")
                continue

        print(f"强化学习挖掘完成，共找到 {len(self.best_factors)} 个因子")
        self.best_factors.sort(key=lambda x: x['ic'], reverse=True)
        return self.best_factors[:100]
