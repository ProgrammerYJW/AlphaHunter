import numpy as np
import os


class Config:
    # 数据文件配置
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    # 时间段配置
    TIME_PERIODS = [
        # {
        #     'name': '2010-2014',
        #     'start_date': '2010-01-01',
        #     'end_date': '2014-12-31',
        #     'file_pattern': '20100101-20141231 P*.csv'
        # },
        # {
        #     'name': '2015-2019',
        #     'start_date': '2015-01-01',
        #     'end_date': '2019-12-31',
        #     'file_pattern': '20150101-20191231 P*.csv'
        # },
        # {
        #     'name': '2020-2023',
        #     'start_date': '2020-01-01',
        #     'end_date': '2023-12-31',
        #     'file_pattern': '20200101-20231231 P*.csv'
        # },
        {
            'name': '2025-2025',
            'start_date': '2025-01-01',
            'end_date': '2025-04-01',
            'file_pattern': '20250101-20250401 P*.csv'
        }
    ]

    # 数据时间范围
    START_DATE = "2025-01-01"
    END_DATE = "2025-04-01"

    # 股票池配置（可选）
    UNIVERSE = []  # 空列表表示使用所有股票

    # 因子挖掘通用参数
    MAX_FACTOR_LENGTH = 20

    GP_POPULATION_SIZE = 300  # 增加种群大小
    GP_GENERATIONS = 50  # 增加进化代数
    GP_CROSSOVER_PROB = 0.85  # 提高交叉概率
    GP_MUTATION_PROB = 0.15  # 提高变异概率
    GP_TOURNAMENT_SIZE = 5  # 增加锦标赛大小
    GP_MIN_TREE_HEIGHT = 3  # 调整树高度范围
    GP_MAX_TREE_HEIGHT = 8

    # 因子筛选标准（调整为更严格）
    MIN_IC = 0.03  # 提高最小IC阈值
    MIN_ICIR = 0.75  # 提高最小ICIR阈值
    MAX_DRAWDOWN = 0.15  # 降低最大回撤阈值

    # 强化学习参数
    RL_LEARNING_RATE = 0.0009
    RL_GAMMA = 0.99
    RL_EPISODES = 1000
    RL_STATE_DIM = 100
    RL_HIDDEN_DIM = 128

    # 数据预处理参数
    WINDOW_SIZE = 55
    FORWARD_DAYS = 5