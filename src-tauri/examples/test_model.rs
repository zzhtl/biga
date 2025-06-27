use biga_lib::stock_prediction::candle_prediction::{ModelConfig, TrainingRequest};
use candle_core::Device;
use chrono::NaiveDate;

#[tokio::main]
async fn main() {
    println!("测试股票预测模型创建...");
    
    // 创建一个简单的训练请求
    let request = TrainingRequest {
        stock_code: "603005.SH".to_string(),
        model_name: "测试模型".to_string(),
        start_date: NaiveDate::from_ymd_opt(2022, 1, 1).unwrap().format("%Y-%m-%d").to_string(),
        end_date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap().format("%Y-%m-%d").to_string(),
        features: vec!["close".to_string(), "volume".to_string(), "ma5".to_string()],
        target: "price".to_string(),
        prediction_days: 5,
        model_type: "linear".to_string(),
        epochs: 5,
        batch_size: 32,
        learning_rate: 0.001,
        dropout: 0.2,
        train_test_split: 0.8,
    };
    
    // 导入模型训练函数
    use biga_lib::stock_prediction::candle_prediction::train_candle_model;
    
    // 训练模型
    match train_candle_model(request).await {
        Ok(result) => {
            println!("模型训练成功！");
            println!("模型ID: {}", result.metadata.id);
            println!("准确率: {:.2}%", result.accuracy * 100.0);
        }
        Err(e) => {
            eprintln!("模型训练失败: {}", e);
        }
    }
} 