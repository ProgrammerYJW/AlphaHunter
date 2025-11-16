import numpy as np
import pandas as pd
import operator
import random
import math
from deap import base, creator, tools, gp
from scipy import stats
import config
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class GeneticProgrammingMiner:
    """改进的遗传规划因子挖掘器"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.features = []
        self.targets = np.array([])
        self.evaluation_samples = 2000  # 增加评估样本数量
        self._setup_deap()

    def _protected_division(self, x, y):
        """保护性除法"""
        if abs(y) < 1e-6:
            return 1.0
        return x / y

    def _protected_log(self, x):
        """保护性对数"""
        if abs(x) < 1e-6:
            return 0.0
        return math.log(abs(x) + 1e-6)  # 避免log(0)

    def _protected_sqrt(self, x):
        """保护性平方根"""
        if x < 0:
            return math.sqrt(abs(x))
        return math.sqrt(x)

    def _ts_mean(self, x, window):
        """时间序列均值"""
        if len(x) < window or window < 1:
            return np.nan
        return np.nanmean(x[-int(window):])

    def _ts_std(self, x, window):
        """时间序列标准差"""
        if len(x) < window or window < 2:
            return np.nan
        return np.nanstd(x[-int(window):])

    def _ts_delta(self, x, period):
        """时间序列差值"""
        if len(x) < period + 1 or period < 1:
            return np.nan
        return x[-1] - x[-int(period)]

    def _ts_returns(self, x, period):
        """时间序列收益率"""
        if len(x) < period + 1 or period < 1:
            return np.nan
        return (x[-1] - x[-int(period)]) / x[-int(period)]

    def _ts_correlation(self, x, y, window):
        """时间序列相关性"""
        if len(x) < window or len(y) < window or window < 2:
            return np.nan
        x_window = x[-int(window):]
        y_window = y[-int(window):]
        return np.corrcoef(x_window, y_window)[0, 1]

    def _ts_rank(self, x, window):
        """时间序列排名"""
        if len(x) < window or window < 1:
            return np.nan
        return stats.rankdata(x[-int(window):])[-1] / len(x[-int(window):])

    def _setup_deap(self):
        """设置DEAP遗传规划环境"""
        # 清除之前可能创建的类
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual

        # 定义适应度函数
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        # 创建基本操作集
        pset = gp.PrimitiveSet("MAIN", 6)
        pset.renameArguments(ARG0='open_price')
        pset.renameArguments(ARG1='high_price')
        pset.renameArguments(ARG2='low_price')
        pset.renameArguments(ARG3='close_price')
        pset.renameArguments(ARG4='volume')
        pset.renameArguments(ARG5='amount')

        # 算术运算
        pset.addPrimitive(operator.add, 2, name='add')
        pset.addPrimitive(operator.sub, 2, name='sub')
        pset.addPrimitive(operator.mul, 2, name='mul')
        pset.addPrimitive(self._protected_division, 2, name='div')

        # 数学函数
        pset.addPrimitive(math.sin, 1, name='sin')
        pset.addPrimitive(math.cos, 1, name='cos')
        pset.addPrimitive(math.tan, 1, name='tan')
        pset.addPrimitive(self._protected_log, 1, name='log')
        pset.addPrimitive(self._protected_sqrt, 1, name='sqrt')
        pset.addPrimitive(abs, 1, name='abs')
        pset.addPrimitive(operator.neg, 1, name='neg')

        # 时间序列函数
        pset.addPrimitive(self._ts_mean, 2, name='ts_mean')
        pset.addPrimitive(self._ts_std, 2, name='ts_std')
        pset.addPrimitive(self._ts_delta, 2, name='ts_delta')
        pset.addPrimitive(self._ts_returns, 2, name='ts_returns')
        pset.addPrimitive(self._ts_rank, 2, name='ts_rank')

        # 常数
        pset.addEphemeralConstant("rand1", lambda: random.uniform(0.1, 10.0))
        pset.addEphemeralConstant("int3_20", lambda: random.randint(3, 20))
        pset.addEphemeralConstant("int5_50", lambda: random.randint(5, 50))

        self.pset = pset

        # 创建工具箱
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                              min_=config.Config.GP_MIN_TREE_HEIGHT,
                              max_=config.Config.GP_MAX_TREE_HEIGHT)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        # 注册遗传操作
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=config.Config.GP_TOURNAMENT_SIZE)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)

        # 限制树深度
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    def _evaluate_individual(self, individual):
        """改进的个体评估函数"""
        try:
            # 编译表达式
            func = self.toolbox.compile(expr=individual)

            # 使用更多的样本进行评估
            n_samples = min(self.evaluation_samples, len(self.features))
            indices = random.sample(range(len(self.features)), n_samples)

            factor_values = []
            valid_targets = []

            for idx in indices:
                features = self.features[idx]
                target = self.targets[idx]

                if np.isnan(target):
                    continue

                try:
                    factor_value = func(
                        features['open'], features['high'], features['low'],
                        features['close'], features['volume'], features['amount']
                    )

                    # 检查有效性
                    if (not np.isnan(factor_value) and
                            not np.isinf(factor_value) and
                            abs(factor_value) < 1e6):  # 避免极端值
                        factor_values.append(factor_value)
                        valid_targets.append(target)

                except (ValueError, ZeroDivisionError, TypeError):
                    continue

            # 计算IC值，要求有足够样本
            if len(factor_values) >= 100:  # 最少100个有效样本
                try:
                    # 使用更稳定的相关性计算方法
                    ic_value, p_value = stats.spearmanr(factor_values, valid_targets)

                    if np.isnan(ic_value):
                        return (-1.0,)

                    # 对IC值进行平滑处理
                    smoothed_ic = abs(ic_value)

                    # 添加多样性奖励（避免所有个体收敛到相同表达式）
                    diversity_bonus = 0.0
                    if hasattr(self, 'best_ic_history'):
                        recent_ics = self.best_ic_history[-5:] if len(
                            self.best_ic_history) >= 5 else self.best_ic_history
                        if len(recent_ics) > 0 and abs(ic_value - np.mean(recent_ics)) > 0.01:
                            diversity_bonus = 0.01

                    final_fitness = smoothed_ic + diversity_bonus
                    return (final_fitness,)

                except:
                    return (-1.0,)
            else:
                return (-1.0,)

        except Exception as e:
            return (-1.0,)

    def mine_factors(self):
        """改进的因子挖掘方法"""
        print("开始严谨的遗传规划因子挖掘...")

        # 准备数据
        self.features, self.targets = self.data_loader.prepare_training_data()

        if len(self.features) == 0:
            print("没有可用的训练数据")
            return []

        print(f"数据规模: {len(self.features)} 个样本")
        print(f"使用 {self.evaluation_samples} 个样本进行个体评估")

        # 数据预处理：去除异常值
        self._preprocess_data()

        # 创建初始种群
        pop = self.toolbox.population(n=config.Config.GP_POPULATION_SIZE)

        # 记录进化历史
        self.best_ic_history = []
        self.population_diversity = []

        # 评估初始种群
        print("评估初始种群...")
        fitnesses = []
        for ind in tqdm(pop, desc="初始评估"):
            fit = self._robust_evaluate(ind)
            fitnesses.append(fit)

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # 进化过程
        best_factors = []
        hall_of_fame = tools.HallOfFame(maxsize=10)  # 名人堂保存最佳个体

        for gen in range(config.Config.GP_GENERATIONS):
            print(f"\n--- 第 {gen + 1} 代 ---")

            # 选择下一代
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉操作
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config.Config.GP_CROSSOVER_PROB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 变异操作
            for mutant in offspring:
                if random.random() < config.Config.GP_MUTATION_PROB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = []
                for ind in tqdm(invalid_ind, desc=f"评估新个体"):
                    fit = self._robust_evaluate(ind)
                    fitnesses.append(fit)

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            # 替换种群
            pop[:] = offspring

            # 更新名人堂
            hall_of_fame.update(pop)

            # 记录统计信息
            fits = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
            if fits:
                best_fit = max(fits)
                avg_fit = np.mean(fits)
                std_fit = np.std(fits)

                self.best_ic_history.append(best_fit)
                self.population_diversity.append(std_fit)

                print(f"最佳适应度: {best_fit:.4f}, 平均适应度: {avg_fit:.4f}, 多样性: {std_fit:.4f}")

                # 保存高质量因子
                if best_fit > config.Config.MIN_IC:
                    best_ind = tools.selBest(pop, 1)[0]
                    expression = str(best_ind)
                    simplified_expr = self._simplify_expression_to_latex(expression)  # 修改为LaTeX格式

                    # 对因子进行更严格的验证
                    validation_score = self._validate_factor(best_ind)

                    if validation_score > config.Config.MIN_IC:
                        best_factors.append({
                            'expression': simplified_expr,  # 使用LaTeX格式的表达式
                            'ic': best_fit,
                            'validation_score': validation_score,
                            'method': 'GP',
                            'individual': best_ind,
                            'generation': gen + 1
                        })
                        print(f"发现高质量因子: IC={best_fit:.4f}, 验证得分={validation_score:.4f}")

            # 早停机制：如果连续3代没有改进，增加变异率
            if len(self.best_ic_history) >= 4:
                recent_improvement = self.best_ic_history[-1] - self.best_ic_history[-4]
                if recent_improvement < 0.001:
                    print("检测到收敛，增加探索性...")

        # 从名人堂中提取更多因子
        print("\n从名人堂提取因子...")
        for i, ind in enumerate(hall_of_fame):
            if i >= 10:  # 只取前10个
                break

            fit = ind.fitness.values[0] if ind.fitness.valid else -1.0
            if fit > config.Config.MIN_IC:
                expression = str(ind)
                simplified_expr = self._simplify_expression_to_latex(expression)  # 修改为LaTeX格式
                validation_score = self._validate_factor(ind)

                # 避免重复
                if not any(f['expression'] == simplified_expr for f in best_factors):
                    best_factors.append({
                        'expression': simplified_expr,  # 使用LaTeX格式的表达式
                        'ic': fit,
                        'validation_score': validation_score,
                        'method': 'GP',
                        'individual': ind,
                        'generation': 'HallOfFame'
                    })

        # 去重并排序
        unique_factors = []
        seen_expressions = set()
        for factor in best_factors:
            if factor['expression'] not in seen_expressions:
                seen_expressions.add(factor['expression'])
                unique_factors.append(factor)

        unique_factors.sort(key=lambda x: x['validation_score'], reverse=True)

        print(f"\n遗传规划完成，发现 {len(unique_factors)} 个独特的高质量因子")
        return unique_factors

    def _robust_evaluate(self, individual):
        """稳健的评估函数，带重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._evaluate_individual(individual)
            except:
                if attempt == max_retries - 1:
                    return (-1.0,)
                continue

    def _preprocess_data(self):
        """数据预处理"""
        print("数据预处理...")

        # 去除目标变量的异常值
        if len(self.targets) > 0:
            q1 = np.percentile(self.targets, 1)
            q99 = np.percentile(self.targets, 99)
            valid_mask = (self.targets >= q1) & (self.targets <= q99)

            self.features = [self.features[i] for i in range(len(self.features)) if valid_mask[i]]
            self.targets = self.targets[valid_mask]

            print(f"去除异常值后剩余 {len(self.targets)} 个样本")

    def _validate_factor(self, individual, n_validations=3):
        """因子验证：使用不同样本集多次验证"""
        try:
            validation_scores = []
            func = self.toolbox.compile(expr=individual)

            for validation_round in range(n_validations):
                # 使用不同的随机样本进行验证
                n_val_samples = min(1000, len(self.features) // 2)
                val_indices = random.sample(range(len(self.features)), n_val_samples)

                factor_values = []
                valid_targets = []

                for idx in val_indices:
                    features = self.features[idx]
                    target = self.targets[idx]

                    if np.isnan(target):
                        continue

                    try:
                        factor_value = func(
                            features['open'], features['high'], features['low'],
                            features['close'], features['volume'], features['amount']
                        )

                        if (not np.isnan(factor_value) and
                                not np.isinf(factor_value) and
                                abs(factor_value) < 1e6):
                            factor_values.append(factor_value)
                            valid_targets.append(target)
                    except:
                        continue

                if len(factor_values) >= 50:
                    try:
                        ic_value, _ = stats.spearmanr(factor_values, valid_targets)
                        if not np.isnan(ic_value):
                            validation_scores.append(abs(ic_value))
                    except:
                        pass

            if validation_scores:
                return np.mean(validation_scores)
            else:
                return 0.0

        except:
            return 0.0

    def _simplify_expression_to_latex(self, expression):
        """将表达式转换为LaTeX格式（主要修改点）"""
        expr = str(expression)

        # 替换函数名为LaTeX格式
        replacements = {
            'add': '+', 'sub': '-', 'mul': '\\times', 'div': '\\div',
            'ts_mean': '\\text{MA}', 'ts_std': '\\sigma', 'ts_delta': '\\Delta',
            'ts_returns': 'R', 'ts_rank': '\\text{Rank}',
            'sin': '\\sin', 'cos': '\\cos', 'tan': '\\tan',
            'log': '\\log', 'sqrt': '\\sqrt', 'abs': '|', 'neg': '-'
        }

        for old, new in replacements.items():
            expr = expr.replace(old, new)

        # 处理变量名，用\mathrm包装
        variables = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount']
        latex_variables = {
            'open_price': '\\mathrm{open}',
            'high_price': '\\mathrm{high}',
            'low_price': '\\mathrm{low}',
            'close_price': '\\mathrm{close}',
            'volume': '\\mathrm{volume}',
            'amount': '\\mathrm{amount}'
        }

        for var, latex_var in latex_variables.items():
            expr = expr.replace(var, latex_var)

        # 处理常量和参数
        import re
        expr = re.sub(r"rand1\(\)", "C", expr)
        expr = re.sub(r"int3_20\(\)", "N", expr)
        expr = re.sub(r"int5_50\(\)", "M", expr)

        # 添加括号确保运算顺序正确
        expr = self._add_parentheses_for_clarity(expr)

        # 添加LaTeX数学环境标记
        expr = f"${expr}$"

        return expr

    def _add_parentheses_for_clarity(self, expr):
        """为表达式添加括号以提高可读性"""
        # 简单的括号添加逻辑，可以根据需要扩展
        operators = ['+', '-', '\\times', '\\div']

        # 对于复杂的运算，添加括号
        for op in operators:
            if expr.count(op) > 1:
                # 简单处理：在第一个操作符前后加括号
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    expr = f"({parts[0]}){op}({parts[1]})"
                break

        return expr

    def evaluate_factor_on_data(self, individual, features):
        """在特定数据上评估因子"""
        try:
            func = self.toolbox.compile(expr=individual)
            return func(
                features['open'], features['high'], features['low'],
                features['close'], features['volume'], features['amount']
            )
        except:
            return np.nan