// 策略研究功能的主JavaScript文件
class StrategyResearchApp {
    constructor() {
        this.strategies = [];
        this.currentStrategy = null;
        this.backtestResults = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadStrategyTemplates();
        this.initializeCharts();
    }

    bindEvents() {
        // 策略选择事件
        const strategySelect = document.getElementById('strategySelect');
        if (strategySelect) {
            strategySelect.addEventListener('change', (e) => {
                this.selectStrategy(e.target.value);
            });
        }

        // 回测按钮事件
        const backtestBtn = document.getElementById('backtestBtn');
        if (backtestBtn) {
            backtestBtn.addEventListener('click', () => this.runBacktest());
        }

        // 参数输入事件
        const paramInputs = document.querySelectorAll('.param-input');
        paramInputs.forEach(input => {
            input.addEventListener('change', () => this.updateStrategyParams());
        });
    }

    loadStrategyTemplates() {
        this.strategies = [
            {
                id: 'topk_drop',
                name: 'Top-K Drop策略',
                description: '选择表现最好的K只股票，去除表现最差的N只',
                params: {
                    topK: { name: 'Top K', value: 10, min: 5, max: 50, step: 1 },
                    dropN: { name: 'Drop N', value: 2, min: 0, max: 10, step: 1 },
                    holdingPeriod: { name: '持有期', value: 20, min: 5, max: 60, step: 5 }
                }
            },
            {
                id: 'momentum',
                name: '动量策略',
                description: '基于价格动量选择股票',
                params: {
                    lookback: { name: '回顾期', value: 60, min: 20, max: 120, step: 10 },
                    percentile: { name: '分位数', value: 0.8, min: 0.5, max: 0.95, step: 0.05 },
                    holdingPeriod: { name: '持有期', value: 20, min: 5, max: 60, step: 5 }
                }
            },
            {
                id: 'mean_reversion',
                name: '均值回归策略',
                description: '基于价格偏离均值的程度选择股票',
                params: {
                    lookback: { name: '回顾期', value: 20, min: 10, max: 60, step: 5 },
                    threshold: { name: '阈值', value: 0.02, min: 0.01, max: 0.1, step: 0.01 },
                    holdingPeriod: { name: '持有期', value: 5, min: 3, max: 20, step: 1 }
                }
            },
            {
                id: 'cta',
                name: 'CTA策略',
                description: '商品交易策略，基于技术指标',
                params: {
                    fastMA: { name: '快速均线', value: 10, min: 5, max: 30, step: 1 },
                    slowMA: { name: '慢速均线', value: 30, min: 20, max: 60, step: 5 },
                    atrPeriod: { name: 'ATR周期', value: 14, min: 10, max: 20, step: 1 }
                }
            }
        ];

        this.populateStrategySelect();
    }

    populateStrategySelect() {
        const select = document.getElementById('strategySelect');
        if (select) {
            select.innerHTML = '<option value="">请选择策略</option>';
            this.strategies.forEach(strategy => {
                const option = document.createElement('option');
                option.value = strategy.id;
                option.textContent = strategy.name;
                select.appendChild(option);
            });
        }
    }

    selectStrategy(strategyId) {
        const strategy = this.strategies.find(s => s.id === strategyId);
        if (strategy) {
            this.currentStrategy = strategy;
            this.displayStrategyDetails(strategy);
        }
    }

    displayStrategyDetails(strategy) {
        const detailsDiv = document.getElementById('strategyDetails');
        if (detailsDiv) {
            detailsDiv.innerHTML = `
                <div class="strategy-info">
                    <h3>${strategy.name}</h3>
                    <p>${strategy.description}</p>
                </div>
                <div class="strategy-params">
                    <h4>参数设置</h4>
                    ${Object.entries(strategy.params).map(([key, param]) => `
                        <div class="param-group">
                            <label>${param.name}:</label>
                            <input type="number" 
                                   class="param-input" 
                                   data-param="${key}"
                                   value="${param.value}"
                                   min="${param.min}"
                                   max="${param.max}"
                                   step="${param.step}">
                        </div>
                    `).join('')}
                </div>
            `;
            
            detailsDiv.style.display = 'block';
            
            // 重新绑定参数输入事件
            const paramInputs = detailsDiv.querySelectorAll('.param-input');
            paramInputs.forEach(input => {
                input.addEventListener('change', () => this.updateStrategyParams());
            });
        }
    }

    updateStrategyParams() {
        if (!this.currentStrategy) return;

        const paramInputs = document.querySelectorAll('.param-input');
        paramInputs.forEach(input => {
            const paramName = input.dataset.param;
            if (this.currentStrategy.params[paramName]) {
                this.currentStrategy.params[paramName].value = parseFloat(input.value);
            }
        });
    }

    async runBacktest() {
        if (!this.currentStrategy) {
            this.showMessage('请先选择策略', 'warning');
            return;
        }

        try {
            this.updateUI('running');
            this.showMessage('正在运行回测...', 'info');

            const response = await fetch('/api/strategy/backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    strategy: this.currentStrategy.id,
                    params: this.currentStrategy.params
                })
            });

            if (!response.ok) {
                throw new Error('回测失败');
            }

            const results = await response.json();
            this.backtestResults = results;
            
            this.displayResults(results);
            this.updateUI('complete');
            this.showMessage('回测完成！', 'success');
            
        } catch (error) {
            console.error('回测失败:', error);
            this.showMessage('回测失败: ' + error.message, 'error');
            this.updateUI('error');
        }
    }

    displayResults(results) {
        const resultsDiv = document.getElementById('backtestResults');
        if (!resultsDiv) return;

        resultsDiv.innerHTML = `
            <div class="results-summary">
                <h3>回测结果</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <label>总收益率</label>
                        <value style="color: ${results.totalReturn >= 0 ? '#00c851' : '#ff4444'}">
                            ${(results.totalReturn * 100).toFixed(2)}%
                        </value>
                    </div>
                    <div class="metric">
                        <label>年化收益率</label>
                        <value style="color: ${results.annualReturn >= 0 ? '#00c851' : '#ff4444'}">
                            ${(results.annualReturn * 100).toFixed(2)}%
                        </value>
                    </div>
                    <div class="metric">
                        <label>夏普比率</label>
                        <value>${results.sharpeRatio.toFixed(2)}</value>
                    </div>
                    <div class="metric">
                        <label>最大回撤</label>
                        <value style="color: #ff4444">
                            ${(results.maxDrawdown * 100).toFixed(2)}%
                        </value>
                    </div>
                    <div class="metric">
                        <label>胜率</label>
                        <value>${(results.winRate * 100).toFixed(2)}%</value>
                    </div>
                    <div class="metric">
                        <label>交易次数</label>
                        <value>${results.tradeCount}</value>
                    </div>
                </div>
            </div>
            <div class="charts-section">
                <canvas id="performanceChart" width="800" height="400"></canvas>
            </div>
        `;

        resultsDiv.style.display = 'block';
        
        // 绘制图表
        this.drawPerformanceChart(results);
    }

    drawPerformanceChart(results) {
        const canvas = document.getElementById('performanceChart');
        if (!canvas || !results.equityCurve) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // 清空画布
        ctx.clearRect(0, 0, width, height);

        // 绘制背景
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);

        // 绘制网格
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            const y = (height - 40) * i / 10 + 20;
            ctx.beginPath();
            ctx.moveTo(40, y);
            ctx.lineTo(width - 40, y);
            ctx.stroke();
        }

        // 绘制净值曲线
        if (results.equityCurve && results.equityCurve.length > 0) {
            const minValue = Math.min(...results.equityCurve);
            const maxValue = Math.max(...results.equityCurve);
            const valueRange = maxValue - minValue;

            ctx.strokeStyle = results.totalReturn >= 0 ? '#00c851' : '#ff4444';
            ctx.lineWidth = 2;
            ctx.beginPath();

            results.equityCurve.forEach((value, index) => {
                const x = 40 + (width - 80) * index / (results.equityCurve.length - 1);
                const y = height - 40 - (height - 80) * (value - minValue) / valueRange;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });

            ctx.stroke();
        }

        // 绘制标题
        ctx.fillStyle = '#f2f2f2';
        ctx.font = '14px Arial';
        ctx.fillText('策略净值曲线', width / 2 - 50, 15);
    }

    initializeCharts() {
        // 初始化图表容器
        const chartsSection = document.querySelector('.charts-section');
        if (!chartsSection) {
            const resultsDiv = document.getElementById('backtestResults');
            if (resultsDiv) {
                const chartsDiv = document.createElement('div');
                chartsDiv.className = 'charts-section';
                chartsDiv.innerHTML = '<canvas id="performanceChart" width="800" height="400"></canvas>';
                resultsDiv.appendChild(chartsDiv);
            }
        }
    }

    updateUI(state) {
        const backtestBtn = document.getElementById('backtestBtn');
        
        switch (state) {
            case 'running':
                if (backtestBtn) {
                    backtestBtn.textContent = '回测中...';
                    backtestBtn.disabled = true;
                    backtestBtn.style.opacity = '0.6';
                }
                break;
            case 'complete':
            case 'error':
                if (backtestBtn) {
                    backtestBtn.textContent = '运行回测';
                    backtestBtn.disabled = false;
                    backtestBtn.style.opacity = '1';
                }
                break;
        }
    }

    showMessage(message, type = 'info') {
        let messageDiv = document.getElementById('message');
        if (!messageDiv) {
            messageDiv = document.createElement('div');
            messageDiv.id = 'message';
            messageDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 20px;
                border-radius: 4px;
                color: white;
                font-weight: 500;
                z-index: 1000;
                transition: all 0.3s ease;
            `;
            document.body.appendChild(messageDiv);
        }

        const colors = {
            info: '#4facfe',
            success: '#00c851',
            warning: '#ffbb33',
            error: '#ff4444'
        };

        messageDiv.style.backgroundColor = colors[type] || colors.info;
        messageDiv.textContent = message;
        messageDiv.style.opacity = '1';

        setTimeout(() => {
            messageDiv.style.opacity = '0';
        }, 3000);
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.strategyApp = new StrategyResearchApp();
});