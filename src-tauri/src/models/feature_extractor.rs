use crate::db::models::HistoricalData;
use anyhow::{Result, Context};
use std::collections::HashMap;

pub struct FeatureExtractor {
    // 存储归一化参数
    normalization_params: HashMap<String, (f64, f64)>, // (min, max)
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            normalization_params: HashMap::new(),
        }
    }

    /// 从股票数据中提取特征
    pub fn extract_features(
        &self,
        stock_data: &[HistoricalData],
        features: &[String],
        target: &str,
        prediction_days: u32,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        if stock_data.is_empty() {
            return Err(anyhow::anyhow!("股票数据为空"));
        }

        let mut feature_vectors = Vec::new();
        let mut targets = Vec::new();

        // 确保有足够的数据进行预测
        let min_data_points = prediction_days as usize + 20; // 最少需要20天历史数据
        if stock_data.len() < min_data_points {
            return Err(anyhow::anyhow!("数据不足，需要至少{}天数据", min_data_points));
        }

        // 为每个数据点提取特征
        for i in 20..stock_data.len() - prediction_days as usize {
            let mut feature_vector = Vec::new();
            
            // 提取指定的特征
            for feature_name in features {
                let feature_value = self.calculate_feature(&stock_data, i, feature_name)?;
                feature_vector.push(feature_value);
            }

            // 计算目标值
            let target_value = self.calculate_target(&stock_data, i, target, prediction_days)?;
            
            feature_vectors.push(feature_vector);
            targets.push(target_value);
        }

        Ok((feature_vectors, targets))
    }

    /// 提取用于预测的特征（不需要目标值）
    pub fn extract_prediction_features(
        &self,
        stock_data: &[HistoricalData],
        features: &[String],
        prediction_days: u32,
    ) -> Result<Vec<Vec<f64>>> {
        if stock_data.is_empty() {
            return Err(anyhow::anyhow!("股票数据为空"));
        }

        if stock_data.len() < 20 {
            return Err(anyhow::anyhow!("数据不足，需要至少20天历史数据"));
        }

        let mut feature_vectors = Vec::new();
        
        // 只为最新的数据点提取特征
        let latest_index = stock_data.len() - 1;
        let mut feature_vector = Vec::new();
        
        for feature_name in features {
            let feature_value = self.calculate_feature(&stock_data, latest_index, feature_name)?;
            feature_vector.push(feature_value);
        }
        
        feature_vectors.push(feature_vector);
        Ok(feature_vectors)
    }

    /// 计算单个特征值
    fn calculate_feature(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        feature_name: &str,
    ) -> Result<f64> {
        match feature_name {
            "close" => Ok(stock_data[index].close),
            "open" => Ok(stock_data[index].open),
            "high" => Ok(stock_data[index].high),
            "low" => Ok(stock_data[index].low),
            "volume" => Ok(stock_data[index].volume as f64),
            "change_percent" => Ok(stock_data[index].change_percent),
            "ma5" => self.calculate_moving_average(stock_data, index, 5),
            "ma10" => self.calculate_moving_average(stock_data, index, 10),
            "ma20" => self.calculate_moving_average(stock_data, index, 20),
            "rsi" => self.calculate_rsi(stock_data, index, 14),
            "macd" => self.calculate_macd(stock_data, index),
            "bollinger" => self.calculate_bollinger_position(stock_data, index, 20),
            "stochastic_k" => self.calculate_stochastic_k(stock_data, index, 14),
            "stochastic_d" => self.calculate_stochastic_d(stock_data, index, 14),
            "momentum" => self.calculate_momentum(stock_data, index, 10),
            _ => Err(anyhow::anyhow!("未知特征: {}", feature_name)),
        }
    }

    /// 计算目标值
    fn calculate_target(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        target: &str,
        prediction_days: u32,
    ) -> Result<f64> {
        let future_index = index + prediction_days as usize;
        if future_index >= stock_data.len() {
            return Err(anyhow::anyhow!("未来数据索引超出范围"));
        }

        match target {
            "close" => {
                let current_price = stock_data[index].close;
                let future_price = stock_data[future_index].close;
                Ok((future_price - current_price) / current_price) // 返回价格变化率
            }
            "change_percent" => Ok(stock_data[future_index].change_percent),
            _ => Err(anyhow::anyhow!("未知目标: {}", target)),
        }
    }

    /// 计算移动平均线
    fn calculate_moving_average(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        period: usize,
    ) -> Result<f64> {
        if index < period - 1 {
            return Ok(stock_data[index].close); // 数据不足时返回当前价格
        }

        let sum: f64 = stock_data[index + 1 - period..=index]
            .iter()
            .map(|d| d.close)
            .sum();
        Ok(sum / period as f64)
    }

    /// 计算RSI
    fn calculate_rsi(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        period: usize,
    ) -> Result<f64> {
        if index < period {
            return Ok(50.0); // 数据不足时返回中性值
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in (index + 1 - period)..=index {
            let change = stock_data[i].change;
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        gains /= period as f64;
        losses /= period as f64;
        
        if losses == 0.0 {
            Ok(100.0)
        } else {
            let rs = gains / losses;
            Ok(100.0 - (100.0 / (1.0 + rs)))
        }
    }

    /// 计算MACD
    fn calculate_macd(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
    ) -> Result<f64> {
        if index < 26 {
            return Ok(0.0);
        }

        let ema12 = self.calculate_ema(stock_data, index, 12)?;
        let ema26 = self.calculate_ema(stock_data, index, 26)?;
        Ok(ema12 - ema26)
    }

    /// 计算EMA
    fn calculate_ema(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        period: usize,
    ) -> Result<f64> {
        if index < period - 1 {
            return Ok(stock_data[index].close);
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = stock_data[index + 1 - period].close;

        for i in (index + 2 - period)..=index {
            ema = (stock_data[i].close - ema) * multiplier + ema;
        }

        Ok(ema)
    }

    /// 计算布林带位置
    fn calculate_bollinger_position(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        period: usize,
    ) -> Result<f64> {
        if index < period - 1 {
            return Ok(0.0);
        }

        let ma = self.calculate_moving_average(stock_data, index, period)?;
        let prices: Vec<f64> = stock_data[index + 1 - period..=index]
            .iter()
            .map(|d| d.close)
            .collect();
        
        let variance: f64 = prices.iter()
            .map(|&price| (price - ma).powi(2))
            .sum::<f64>() / period as f64;
        
        let std_dev = variance.sqrt();
        let upper_band = ma + 2.0 * std_dev;
        let lower_band = ma - 2.0 * std_dev;

        if upper_band == lower_band {
            Ok(0.0)
        } else {
            Ok((stock_data[index].close - lower_band) / (upper_band - lower_band) - 0.5)
        }
    }

    /// 计算随机指标K值
    fn calculate_stochastic_k(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        period: usize,
    ) -> Result<f64> {
        if index < period - 1 {
            return Ok(0.5);
        }

        let data_slice = &stock_data[index + 1 - period..=index];
        let highest_high = data_slice.iter().map(|d| d.high).fold(f64::NEG_INFINITY, f64::max);
        let lowest_low = data_slice.iter().map(|d| d.low).fold(f64::INFINITY, f64::min);

        if highest_high == lowest_low {
            Ok(0.5)
        } else {
            Ok((stock_data[index].close - lowest_low) / (highest_high - lowest_low))
        }
    }

    /// 计算随机指标D值
    fn calculate_stochastic_d(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        period: usize,
    ) -> Result<f64> {
        if index < period + 2 {
            return Ok(0.5);
        }

        // 计算过去3天的K值并取平均
        let mut k_values = Vec::new();
        for i in 0..3 {
            if index >= i {
                let k = self.calculate_stochastic_k(stock_data, index - i, period)?;
                k_values.push(k);
            }
        }

        if k_values.is_empty() {
            Ok(0.5)
        } else {
            Ok(k_values.iter().sum::<f64>() / k_values.len() as f64)
        }
    }

    /// 计算动量指标
    fn calculate_momentum(
        &self,
        stock_data: &[HistoricalData],
        index: usize,
        period: usize,
    ) -> Result<f64> {
        if index < period {
            return Ok(0.0);
        }

        let current_price = stock_data[index].close;
        let past_price = stock_data[index - period].close;
        Ok(current_price / past_price - 1.0)
    }

    /// 归一化特征
    pub fn normalize_features(&mut self, features: &mut Vec<Vec<f64>>) -> Result<()> {
        if features.is_empty() {
            return Ok(());
        }

        let feature_count = features[0].len();
        
        for feature_idx in 0..feature_count {
            let values: Vec<f64> = features.iter().map(|row| row[feature_idx]).collect();
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            self.normalization_params.insert(
                format!("feature_{}", feature_idx),
                (min_val, max_val)
            );

            // 归一化到[0, 1]区间
            for row in features.iter_mut() {
                if max_val != min_val {
                    row[feature_idx] = (row[feature_idx] - min_val) / (max_val - min_val);
                } else {
                    row[feature_idx] = 0.0;
                }
            }
        }
        
        Ok(())
    }
    
    /// 归一化目标值
    pub fn normalize_targets(&mut self, targets: &mut Vec<f64>) -> Result<(f64, f64)> {
        if targets.is_empty() {
            return Ok((0.0, 1.0));
        }

        let min_val = targets.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = targets.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range != 0.0 {
            for target in targets.iter_mut() {
                *target = (*target - min_val) / range;
            }
        }

        Ok((min_val, range))
    }

    /// 反归一化预测结果
    pub fn denormalize_prediction(&self, prediction: f64, min_value: f64, range: f64) -> f64 {
            prediction * range + min_value
    }
        }

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
} 