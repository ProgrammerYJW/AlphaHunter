// 因子挖掘功能的主JavaScript文件
class FactorMiningApp {
    constructor() {
        this.eventSource = null;
        this.isRunning = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadPreviousResults();
    }

    bindEvents() {
        const startBtn = document.getElementById('startBtn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startMining());
        }

        // 监听算法选择变化
        const algorithmRadios = document.querySelectorAll('input[name="algorithm"]');
        algorithmRadios.forEach(radio => {
            radio.addEventListener('change', () => this.updateButtonText());
        });
    }

    updateButtonText() {
        const startBtn = document.getElementById('startBtn');
        const selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked').value;
        
        if (startBtn) {
            const buttonText = startBtn.querySelector('i').nextSibling;
            switch(selectedAlgorithm) {
                case 'gp':
                    buttonText.textContent = ' 开始挖掘(遗传算法)';
                    break;
                case 'rl':
                    buttonText.textContent = ' 开始挖掘(强化学习)';
                    break;
                case 'hybrid':
                    buttonText.textContent = ' 开始挖掘(混合算法)';
                    break;
            }
        }
    }

    async startMining() {
        if (this.isRunning) {
            this.showMessage('因子挖掘正在进行中...', 'warning');
            return;
        }

        const algorithm = document.querySelector('input[name="algorithm"]:checked')?.value || 'gp';
        
        try {
            this.isRunning = true;
            this.updateUI('running');
            this.clearLog();
            this.showMessage('正在启动因子挖掘...', 'info');

            // 调用后端API开始挖掘
            const response = await fetch('/api/factor/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ algorithm: algorithm })
            });

            if (!response.ok) {
                throw new Error('启动失败');
            }

            // 开始接收实时日志
            this.startLogStream();
            
        } catch (error) {
            console.error('启动因子挖掘失败:', error);
            this.showMessage('启动失败: ' + error.message, 'error');
            this.isRunning = false;
            this.updateUI('error');
        }
    }

    startLogStream() {
        if (this.eventSource) {
            this.eventSource.close();
        }

        this.eventSource = new EventSource('/api/factor/stream');
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleStreamData(data);
            } catch (error) {
                console.error('解析流数据失败:', error);
            }
        };

        this.eventSource.onerror = (error) => {
            console.error('EventSource错误:', error);
            if (this.isRunning) {
                this.showMessage('连接中断，正在重试...', 'warning');
                setTimeout(() => {
                    if (this.isRunning) {
                        this.startLogStream();
                    }
                }, 2000);
            }
        };
    }

    handleStreamData(data) {
        if (data.type === 'ping') {
            return; // 心跳包
        }

        if (data.type === 'log') {
            this.appendLog(data.payload);
        } else if (data.type === 'progress') {
            this.updateProgress(data.payload);
        } else if (data.type === 'factor') {
            this.addFactorResult(data.payload);
        } else if (data.type === 'done') {
            this.onMiningComplete();
        } else if (data.type === 'error') {
            this.showMessage(data.payload, 'error');
            this.isRunning = false;
            this.updateUI('error');
        }
    }

    appendLog(message) {
        const logDiv = document.getElementById('log');
        if (logDiv) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.innerHTML = `<span style="color: #4facfe;">[${timestamp}]</span> ${message}`;
            logDiv.appendChild(logEntry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
    }

    clearLog() {
        const logDiv = document.getElementById('log');
        if (logDiv) {
            logDiv.innerHTML = '';
        }
    }

    updateProgress(progress) {
        // 可以添加进度条显示
        const progressInfo = `进度: ${progress.current}/${progress.total} (${progress.percentage}%)`;
        this.appendLog(progressInfo);
    }

    addFactorResult(factor) {
        const resultDiv = document.getElementById('result');
        if (resultDiv) {
            resultDiv.style.display = 'block';
            
            const tbody = resultDiv.querySelector('tbody');
            if (tbody) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${tbody.children.length + 1}</td>
                    <td>${this.formatLatex(factor.expression)}</td>
                    <td>${factor.ic.toFixed(4)}</td>
                    <td>${factor.icir.toFixed(4)}</td>
                    <td>${factor.score.toFixed(4)}</td>
                    <td>
                        <button class="copy-btn" onclick="copyToClipboard('${factor.expression}')">
                            复制
                        </button>
                    </td>
                `;
                tbody.appendChild(row);
            }
        }
    }

    formatLatex(latex) {
        // 简单的LaTeX格式化
        return latex.replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '$1/$2')
                   .replace(/\\times/g, '×')
                   .replace(/\\div/g, '÷')
                   .replace(/\\alpha/g, 'α')
                   .replace(/\\beta/g, 'β')
                   .replace(/\\gamma/g, 'γ');
    }

    onMiningComplete() {
        this.isRunning = false;
        this.updateUI('complete');
        this.showMessage('因子挖掘完成！', 'success');
        
        // 关闭EventSource
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        // 加载最终结果
        this.loadFinalResults();
    }

    async loadFinalResults() {
        try {
            const response = await fetch('/api/factor/result');
            const results = await response.json();
            
            if (results && results.length > 0) {
                results.forEach((factor, index) => {
                    this.addFactorResult(factor);
                });
            }
        } catch (error) {
            console.error('加载最终结果失败:', error);
        }
    }

    async loadPreviousResults() {
        try {
            const response = await fetch('/api/factor/result');
            const results = await response.json();
            
            if (results && results.length > 0) {
                const resultDiv = document.getElementById('result');
                if (resultDiv) {
                    resultDiv.style.display = 'block';
                    
                    const tbody = resultDiv.querySelector('tbody');
                    if (tbody) {
                        tbody.innerHTML = '';
                        results.forEach((factor, index) => {
                            this.addFactorResult(factor);
                        });
                    }
                }
            }
        } catch (error) {
            console.log('没有历史结果');
        }
    }

    updateUI(state) {
        const startBtn = document.getElementById('startBtn');
        
        switch (state) {
            case 'running':
                if (startBtn) {
                    startBtn.textContent = '挖掘中...';
                    startBtn.disabled = true;
                    startBtn.style.opacity = '0.6';
                }
                break;
            case 'complete':
            case 'error':
                if (startBtn) {
                    // 恢复按钮文本为当前选择的算法
                    const selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked')?.value || 'gp';
                    let buttonText = ' 开始挖掘';
                    switch(selectedAlgorithm) {
                        case 'gp':
                            buttonText = ' 开始挖掘(遗传算法)';
                            break;
                        case 'rl':
                            buttonText = ' 开始挖掘(强化学习)';
                            break;
                        case 'hybrid':
                            buttonText = ' 开始挖掘(混合算法)';
                            break;
                    }
                    
                    // 重新设置按钮内容，保留图标
                    const icon = startBtn.querySelector('i');
                    startBtn.innerHTML = '';
                    startBtn.appendChild(icon);
                    startBtn.appendChild(document.createTextNode(buttonText));
                    
                    startBtn.disabled = false;
                    startBtn.style.opacity = '1';
                }
                break;
        }
    }

    showMessage(message, type = 'info') {
        // 创建消息提示
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

// 工具函数
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('已复制到剪贴板: ' + text);
    }).catch(err => {
        console.error('复制失败:', err);
        alert('复制失败');
    });
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.factorApp = new FactorMiningApp();
});