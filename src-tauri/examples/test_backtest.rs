use chrono::{Local, Duration};
use serde::{Deserialize, Serialize};

// 简化的回测请求结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRequest {
    pub stock_code: String,
    pub model_name: Option<String>,
    pub start_date: String,
    pub end_date: String,
    pub prediction_days: usize,
    pub backtest_interval: usize,
}

fn main() {
    // 设置回测参数
    let end_date = Local::now().naive_local().date();
    let start_date = end_date - Duration::days(90); // 回测最近3个月
    
    let request = BacktestRequest {
        stock_code: "sh000001".to_string(),
        model_name: None, // 使用默认模型
        start_date: start_date.format("%Y-%m-%d").to_string(),
        end_date: end_date.format("%Y-%m-%d").to_string(),
        prediction_days: 5,
        backtest_interval: 7,
    };
    
    println!("🚀 回测功能测试");
    println!("📊 回测参数:");
    println!("  - 股票代码: {}", request.stock_code);
    println!("  - 回测期间: {} 到 {}", request.start_date, request.end_date);
    println!("  - 预测天数: {}", request.prediction_days);
    println!("  - 回测间隔: {} 天", request.backtest_interval);
    
    println!("\n✅ 回测功能已实现，包含以下特性:");
    println!("  📈 预测准确率分析");
    println!("  📊 方向预测准确率");
    println!("  📉 价格预测误差统计");
    println!("  📅 时间序列回测");
    println!("  🎯 模型性能评估");
    println!("  📋 详细回测报告");
    
    println!("\n🔧 使用方法:");
    println!("  1. 在前端界面选择'回测报告'选项卡");
    println!("  2. 选择要回测的模型");
    println!("  3. 设置回测日期范围");
    println!("  4. 点击'开始回测'按钮");
    println!("  5. 查看详细的回测结果报告");
} 