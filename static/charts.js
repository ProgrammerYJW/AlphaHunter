// 图表库 - 用于绘制各种金融图表
class ChartLibrary {
    constructor() {
        this.colors = {
            primary: '#4facfe',
            secondary: '#00f2fe',
            success: '#00c851',
            danger: '#ff4444',
            warning: '#ffbb33',
            background: '#0e0e0e',
            grid: '#333',
            text: '#f2f2f2'
        };
    }

    // 绘制折线图
    drawLineChart(canvas, data, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // 清空画布
        ctx.clearRect(0, 0, width, height);
        
        // 设置背景
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        // 绘制网格
        this.drawGrid(ctx, width, height);
        
        // 计算数据范围
        const values = data.values || data;
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue;
        
        // 绘制线条
        ctx.strokeStyle = options.color || this.colors.primary;
        ctx.lineWidth = options.lineWidth || 2;
        ctx.beginPath();
        
        values.forEach((value, index) => {
            const x = 40 + (width - 80) * index / (values.length - 1);
            const y = height - 40 - (height - 80) * (value - minValue) / valueRange;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // 绘制标题
        if (options.title) {
            this.drawTitle(ctx, width, options.title);
        }
        
        // 绘制坐标轴标签
        this.drawAxisLabels(ctx, width, height, minValue, maxValue);
    }

    // 绘制面积图
    drawAreaChart(canvas, data, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        this.drawGrid(ctx, width, height);
        
        const values = data.values || data;
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue;
        
        // 创建渐变
        const gradient = ctx.createLinearGradient(0, height - 40, 0, 40);
        gradient.addColorStop(0, options.color || this.colors.primary);
        gradient.addColorStop(1, 'transparent');
        
        // 绘制面积
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(40, height - 40);
        
        values.forEach((value, index) => {
            const x = 40 + (width - 80) * index / (values.length - 1);
            const y = height - 40 - (height - 80) * (value - minValue) / valueRange;
            ctx.lineTo(x, y);
        });
        
        ctx.lineTo(width - 40, height - 40);
        ctx.closePath();
        ctx.fill();
        
        // 绘制边框线
        ctx.strokeStyle = options.borderColor || this.colors.primary;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        values.forEach((value, index) => {
            const x = 40 + (width - 80) * index / (values.length - 1);
            const y = height - 40 - (height - 80) * (value - minValue) / valueRange;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        if (options.title) {
            this.drawTitle(ctx, width, options.title);
        }
        
        this.drawAxisLabels(ctx, width, height, minValue, maxValue);
    }

    // 绘制柱状图
    drawBarChart(canvas, data, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        const values = data.values || data;
        const labels = data.labels || [];
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue;
        
        const barWidth = (width - 80) / values.length * 0.8;
        const barSpacing = (width - 80) / values.length;
        
        values.forEach((value, index) => {
            const barHeight = ((value - minValue) / valueRange) * (height - 80);
            const x = 40 + index * barSpacing + (barSpacing - barWidth) / 2;
            const y = height - 40 - barHeight;
            
            // 绘制柱子
            ctx.fillStyle = options.colors ? options.colors[index] : this.colors.primary;
            ctx.fillRect(x, y, barWidth, barHeight);
            
            // 绘制标签
            if (labels[index]) {
                ctx.fillStyle = this.colors.text;
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(labels[index], x + barWidth / 2, height - 25);
            }
            
            // 绘制数值
            ctx.fillStyle = this.colors.text;
            ctx.font = '10px Arial';
            ctx.fillText(value.toFixed(2), x + barWidth / 2, y - 5);
        });
        
        if (options.title) {
            this.drawTitle(ctx, width, options.title);
        }
    }

    // 绘制饼图
    drawPieChart(canvas, data, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 2 - 40;
        
        const values = data.values || data;
        const labels = data.labels || [];
        const total = values.reduce((sum, value) => sum + value, 0);
        
        let currentAngle = -Math.PI / 2;
        
        values.forEach((value, index) => {
            const sliceAngle = (value / total) * 2 * Math.PI;
            
            // 绘制扇形
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
            ctx.closePath();
            
            ctx.fillStyle = options.colors ? options.colors[index] : this.getColor(index);
            ctx.fill();
            
            ctx.strokeStyle = this.colors.background;
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // 绘制标签
            if (labels[index]) {
                const labelAngle = currentAngle + sliceAngle / 2;
                const labelX = centerX + Math.cos(labelAngle) * (radius * 0.7);
                const labelY = centerY + Math.sin(labelAngle) * (radius * 0.7);
                
                ctx.fillStyle = this.colors.text;
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(labels[index], labelX, labelY);
                
                // 绘制百分比
                const percentage = ((value / total) * 100).toFixed(1);
                ctx.font = '10px Arial';
                ctx.fillText(`${percentage}%`, labelX, labelY + 15);
            }
            
            currentAngle += sliceAngle;
        });
        
        if (options.title) {
            this.drawTitle(ctx, width, options.title);
        }
    }

    // 绘制K线图
    drawCandlestickChart(canvas, data, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        this.drawGrid(ctx, width, height);
        
        const ohlc = data.ohlc || data;
        const highs = ohlc.map(d => d.high);
        const lows = ohlc.map(d => d.low);
        const minPrice = Math.min(...lows);
        const maxPrice = Math.max(...highs);
        const priceRange = maxPrice - minPrice;
        
        const candleWidth = (width - 80) / ohlc.length * 0.8;
        const candleSpacing = (width - 80) / ohlc.length;
        
        ohlc.forEach((candle, index) => {
            const x = 40 + index * candleSpacing + (candleSpacing - candleWidth) / 2;
            const openY = height - 40 - (candle.open - minPrice) / priceRange * (height - 80);
            const closeY = height - 40 - (candle.close - minPrice) / priceRange * (height - 80);
            const highY = height - 40 - (candle.high - minPrice) / priceRange * (height - 80);
            const lowY = height - 40 - (candle.low - minPrice) / priceRange * (height - 80);
            
            const isGreen = candle.close >= candle.open;
            
            // 绘制影线
            ctx.strokeStyle = isGreen ? this.colors.success : this.colors.danger;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x + candleWidth / 2, highY);
            ctx.lineTo(x + candleWidth / 2, lowY);
            ctx.stroke();
            
            // 绘制实体
            ctx.fillStyle = isGreen ? this.colors.success : this.colors.danger;
            const bodyTop = Math.min(openY, closeY);
            const bodyHeight = Math.abs(closeY - openY);
            
            if (isGreen) {
                ctx.fillRect(x, bodyTop, candleWidth, bodyHeight);
            } else {
                ctx.fillRect(x, bodyTop, candleWidth, bodyHeight);
                ctx.strokeStyle = this.colors.danger;
                ctx.lineWidth = 1;
                ctx.strokeRect(x, bodyTop, candleWidth, bodyHeight);
            }
        });
        
        if (options.title) {
            this.drawTitle(ctx, width, options.title);
        }
        
        this.drawAxisLabels(ctx, width, height, minPrice, maxPrice);
    }

    // 绘制网格
    drawGrid(ctx, width, height) {
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;
        
        // 水平网格线
        for (let i = 0; i <= 10; i++) {
            const y = 40 + (height - 80) * i / 10;
            ctx.beginPath();
            ctx.moveTo(40, y);
            ctx.lineTo(width - 40, y);
            ctx.stroke();
        }
        
        // 垂直网格线
        for (let i = 0; i <= 10; i++) {
            const x = 40 + (width - 80) * i / 10;
            ctx.beginPath();
            ctx.moveTo(x, 40);
            ctx.lineTo(x, height - 40);
            ctx.stroke();
        }
    }

    // 绘制标题
    drawTitle(ctx, width, title) {
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(title, width / 2, 25);
    }

    // 绘制坐标轴标签
    drawAxisLabels(ctx, width, height, minValue, maxValue) {
        ctx.fillStyle = this.colors.text;
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        
        // Y轴标签
        for (let i = 0; i <= 10; i++) {
            const value = minValue + (maxValue - minValue) * i / 10;
            const y = height - 40 - (height - 80) * i / 10;
            ctx.fillText(value.toFixed(2), 35, y + 4);
        }
    }

    // 获取颜色
    getColor(index) {
        const colors = [
            this.colors.primary,
            this.colors.secondary,
            this.colors.success,
            this.colors.warning,
            this.colors.danger,
            '#9c27b0',
            '#ff9800',
            '#795548',
            '#607d8b',
            '#e91e63'
        ];
        return colors[index % colors.length];
    }

    // 绘制性能指标仪表盘
    drawPerformanceGauge(canvas, value, maxValue, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 2 - 40;
        
        // 绘制背景弧
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, Math.PI * 0.75, Math.PI * 2.25);
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 20;
        ctx.stroke();
        
        // 绘制数值弧
        const angle = Math.PI * 0.75 + (value / maxValue) * Math.PI * 1.5;
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, this.colors.primary);
        gradient.addColorStop(1, this.colors.secondary);
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, Math.PI * 0.75, angle);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 20;
        ctx.stroke();
        
        // 绘制指针
        const pointerLength = radius * 0.8;
        const pointerX = centerX + Math.cos(angle) * pointerLength;
        const pointerY = centerY + Math.sin(angle) * pointerLength;
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(pointerX, pointerY);
        ctx.strokeStyle = this.colors.text;
        ctx.lineWidth = 4;
        ctx.stroke();
        
        // 绘制中心圆
        ctx.beginPath();
        ctx.arc(centerX, centerY, 10, 0, Math.PI * 2);
        ctx.fillStyle = this.colors.primary;
        ctx.fill();
        
        // 绘制数值
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 24px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(value.toFixed(2), centerX, centerY + 60);
        
        if (options.label) {
            ctx.font = '16px Arial';
            ctx.fillText(options.label, centerX, centerY + 85);
        }
        
        if (options.title) {
            this.drawTitle(ctx, width, options.title);
        }
    }
}

// 创建全局图表库实例
window.chartLib = new ChartLibrary();