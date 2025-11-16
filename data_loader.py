import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import config
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings('ignore')


class MultiFileDataLoader:
    def __init__(self):
        self.data_loaded = False
        self.stock_data = None
        self.features = []
        self.targets = np.array([])
        self.training_data_prepared = False

    def check_data_files(self):
        """检查数据文件是否存在"""
        data_dir = config.Config.DATA_DIR

        print(f"检查数据目录: {data_dir}")

        if not os.path.exists(data_dir):
            print(f"数据目录不存在: {data_dir}")
            print("正在创建数据目录...")
            os.makedirs(data_dir)
            return False

        # 检查每个时间段的文件
        found_periods = []
        for period in config.Config.TIME_PERIODS:
            pattern = os.path.join(data_dir, period['file_pattern'])
            files = glob.glob(pattern)

            if files:
                found_periods.append({
                    'name': period['name'],
                    'files': files,
                    'count': len(files)
                })

        if found_periods:
            print(f"找到 {len(found_periods)} 个时间段的数据文件:")
            for period in found_periods:
                print(f"  - {period['name']}: {period['count']} 个文件")
            return True
        else:
            print("未找到任何数据文件！")
            self._print_download_guide(data_dir)
            return False

    def _print_download_guide(self, data_dir):
        """打印数据下载指南"""
        print("\n" + "=" * 60)
        print("                 数据下载指南")
        print("=" * 60)
        print(f"请将CSV文件放置在: {data_dir}")
        print("\n文件格式要求:")
        print("包含字段: 股票代码, 交易日期, 日开盘价, 日收盘价, 日最高价, 日最低价, 日成交股数, 日成交金额")
        print("=" * 60)

    def load_stock_data(self):
        """加载并合并股票数据"""
        if self.data_loaded and self.stock_data is not None:
            return self.stock_data

        print("加载并合并股票数据...")

        if not self.check_data_files():
            print("数据文件缺失，使用样本数据进行演示")
            return self._generate_sample_data()

        all_data_frames = []

        for period in config.Config.TIME_PERIODS:
            pattern = os.path.join(config.Config.DATA_DIR, period['file_pattern'])
            files = glob.glob(pattern)

            if not files:
                print(f"警告: 未找到 {period['name']} 时间段的数据文件")
                continue

            print(f"处理 {period['name']} 时间段: {len(files)} 个文件")

            for file_path in files:
                try:
                    df = self._load_single_file(file_path)
                    if df is not None and not df.empty:
                        all_data_frames.append(df)
                        print(f"  成功加载: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  加载失败 {os.path.basename(file_path)}: {e}")

        if not all_data_frames:
            print("没有成功加载任何数据，将使用模拟数据")
            return self._generate_sample_data()

        # 合并所有数据
        merged_df = pd.concat(all_data_frames, ignore_index=True)
        merged_df = merged_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # 去重
        merged_df = merged_df.drop_duplicates(subset=['symbol', 'date'])

        # 过滤时间范围
        start_date = pd.to_datetime(config.Config.START_DATE)
        end_date = pd.to_datetime(config.Config.END_DATE)
        merged_df = merged_df[(merged_df['date'] >= start_date) & (merged_df['date'] <= end_date)]

        self.stock_data = merged_df
        self.data_loaded = True

        print(f"数据合并完成: {len(merged_df)} 行, {len(merged_df['symbol'].unique())} 只股票")
        return merged_df

    def _load_single_file(self, file_path):
        """加载单个数据文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'utf-8-sig']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return self._clean_stock_data(df)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        raise Exception(f"无法读取文件 {file_path}")

    def _clean_stock_data(self, df):
        """清洗和格式化股票数据"""
        # 列名映射
        column_mapping = {}
        possible_columns = {
            'symbol': ['Stkcd', '股票代码', 'symbol', 'STKCD'],
            'date': ['Trddt', '交易日期', 'date', 'TRDDT'],
            'open': ['Opnprc', '开盘价', 'open', 'OPNPRC'],
            'high': ['Hiprc', '最高价', 'high', 'HIPRC'],
            'low': ['Loprc', '最低价', 'low', 'LOPRC'],
            'close': ['Clsprc', '收盘价', 'close', 'CLSPRC'],
            'volume': ['Dnshrtrd', '成交量', 'volume', 'DNSHRTRD'],
            'amount': ['Dnvaltrd', '成交额', 'amount', 'DNVALTRD']
        }

        # 识别实际列名
        for standard_name, possible_names in possible_columns.items():
            for name in possible_names:
                if name in df.columns:
                    column_mapping[name] = standard_name
                    break

        # 重命名列
        df = df.rename(columns=column_mapping)

        # 检查必要列
        required_columns = ['symbol', 'date', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告: 文件缺少必要列 {missing_columns}")
            return pd.DataFrame()

        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)

        # 转换数值列
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 过滤有效数据
        df = df.dropna(subset=['date', 'symbol', 'close'])
        return df

    def _generate_sample_data(self):
        """生成样本数据"""
        print("生成样本数据用于演示...")

        symbols = config.Config.UNIVERSE if config.Config.UNIVERSE else ['000001', '000002', '600519', '000858',
                                                                         '601318']
        dates = pd.date_range(start=config.Config.START_DATE, end=config.Config.END_DATE, freq='D')

        data = []
        for symbol in symbols:
            base_price = np.random.uniform(10, 100)

            for i, date in enumerate(dates):
                trend = 0.0002 * i
                noise = np.random.normal(0, 0.02)
                price = base_price * (1 + trend + noise)

                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': price * (1 + np.random.uniform(-0.01, 0.01)),
                    'high': price * (1 + np.random.uniform(0, 0.03)),
                    'low': price * (1 + np.random.uniform(-0.03, 0)),
                    'close': price,
                    'volume': np.random.uniform(1e6, 1e7),
                    'amount': price * np.random.uniform(1e6, 1e7)
                })

        df = pd.DataFrame(data)
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        print(f"生成 {len(symbols)} 只样本股票数据")
        return df

    def prepare_training_data(self, symbols=None):
        """准备训练数据 - 核心方法"""
        print("准备训练数据...")

        # 加载股票数据
        if self.stock_data is None:
            self.stock_data = self.load_stock_data()

        if self.stock_data is None or self.stock_data.empty:
            print("无法获取有效数据")
            self.features = []
            self.targets = np.array([])
            self.training_data_prepared = False
            return self.features, self.targets

        # 获取股票池
        if symbols is None:
            symbols = self.get_stock_universe()

        # 过滤指定股票的数据
        stock_data = self.stock_data[self.stock_data['symbol'].isin(symbols)]

        if stock_data.empty:
            print("没有找到指定股票的数据")
            self.features = []
            self.targets = np.array([])
            self.training_data_prepared = False
            return self.features, self.targets

        # 计算特征和目标变量
        self.features, self.targets = self._calculate_features_and_targets(stock_data)
        self.training_data_prepared = True

        print(f"训练数据准备完成: {len(self.features)} 个样本")
        return self.features, self.targets

    def get_stock_universe(self):
        """获取股票池"""
        if self.stock_data is None:
            self.stock_data = self.load_stock_data()

        if config.Config.UNIVERSE:
            universe = [str(s).zfill(6) for s in config.Config.UNIVERSE]
            existing_symbols = self.stock_data['symbol'].unique()
            universe = [s for s in universe if s in existing_symbols]

            if not universe:
                print("配置的股票池中没有股票在数据中，将使用数据中的所有股票")
                universe = existing_symbols
        else:
            universe = self.stock_data['symbol'].unique()

        print(f"股票池包含 {len(universe)} 只股票")
        return universe

    def _calculate_features_and_targets(self, stock_data):
        """计算特征和目标变量"""
        print("计算特征和目标变量...")

        stock_data = stock_data.sort_values(['symbol', 'date'])
        features_list = []
        targets_list = []

        for symbol in tqdm(stock_data['symbol'].unique(), desc="处理股票"):
            symbol_data = stock_data[stock_data['symbol'] == symbol].copy()

            # 确保数据连续性
            symbol_data = symbol_data.dropna(subset=['close'])
            if len(symbol_data) < config.Config.WINDOW_SIZE + config.Config.FORWARD_DAYS + 5:
                continue

            # 计算未来收益率
            symbol_data['future_return'] = (
                    symbol_data['close'].shift(-config.Config.FORWARD_DAYS) / symbol_data['close'] - 1
            )

            # 创建特征窗口
            for i in range(config.Config.WINDOW_SIZE, len(symbol_data) - config.Config.FORWARD_DAYS):
                window_data = symbol_data.iloc[i - config.Config.WINDOW_SIZE:i]

                feature_set = {
                    'open': window_data['open'].values.astype(np.float64),
                    'high': window_data['high'].values.astype(np.float64),
                    'low': window_data['low'].values.astype(np.float64),
                    'close': window_data['close'].values.astype(np.float64),
                    'volume': window_data['volume'].values.astype(np.float64),
                    'amount': window_data['amount'].values.astype(np.float64),
                    'symbol': symbol,
                    'date': symbol_data.iloc[i]['date']
                }

                features_list.append(feature_set)
                targets_list.append(symbol_data.iloc[i]['future_return'])

        print(f"生成 {len(features_list)} 个训练样本")
        return features_list, np.array(targets_list)

    def get_sample_features_for_evaluation(self, n_samples=1000):
        """获取用于评估的样本特征"""
        if not self.training_data_prepared:
            self.prepare_training_data()

        if len(self.features) == 0:
            return self._create_dummy_features(n_samples)

        # 随机采样
        n_samples = min(n_samples, len(self.features))
        sample_indices = random.sample(range(len(self.features)), n_samples)
        sample_features = [self.features[i] for i in sample_indices]
        sample_targets = self.targets[sample_indices]

        return sample_features, sample_targets

    def _create_dummy_features(self, n_samples):
        """创建虚拟特征"""
        features = []
        targets = np.random.normal(0, 0.1, n_samples)

        for i in range(n_samples):
            features.append({
                'open': np.random.normal(10, 2, 20),
                'high': np.random.normal(12, 2, 20),
                'low': np.random.normal(8, 2, 20),
                'close': np.random.normal(10, 2, 20),
                'volume': np.random.normal(1e6, 1e5, 20),
                'amount': np.random.normal(1e7, 1e6, 20)
            })

        return features, targets