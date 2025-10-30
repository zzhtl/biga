//! 简化的预测接口
//! 
//! 整合金融预测引擎，提供统一的预测API

use crate::stock_prediction::{
    financial_prediction_engine::{FinancialPredictionEngine, FinancialPrediction},
    risk_management::RiskManager,
    technical_indicators_optimized::{calculate_all_indicators, TechnicalIndicators},
    core_weights_simplified::WeightManager,
    database::get_recent_market_data,
    types::{PredictionRequest, PredictionResponse, Prediction},
};
use serde::{Deserialize, Serialize};

/// 专业预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfessionalPredictionResult {
    pub predictions: Vec<Prediction>,
    pub technical_indicators: TechnicalIndicators,
    pub risk_assessment: RiskAssessment,
    pub trading_recommendation: TradingRecommendation,
}

/// 风险评估
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub var_95: f64,           // 95%置信水平的VaR
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub optimal_position: f64,
    pub stop_loss: f64,
    pub take_profits: Vec<f64>,
}

/// 交易建议
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRecommendation {
    pub action: String,
    pub confidence: f64,
    pub reasons: Vec<String>,
    pub risk_level: String,
}

/// 统一预测接口
pub struct UnifiedPredictor {
    engine: FinancialPredictionEngine,
    risk_manager: RiskManager,
    weight_manager: WeightManager,
}

impl UnifiedPredictor {
    pub fn new() -> Self {
        Self {
            engine: FinancialPredictionEngine::new(),
            risk_manager: RiskManager::new(),
            weight_manager: WeightManager::new(),
        }
    }
    
    /// 执行预测
    pub async fn predict(&mut self, request: PredictionRequest) -> Result<ProfessionalPredictionResult, String> {
        // 1. 获取市场数据
        let (current_price, _, dates, prices, volumes, highs, lows) = 
            get_recent_market_data(&request.stock_code, 120).await
            .map_err(|e| format!("获取市场数据失败: {e}"))?;
        
        // 2. 计算技术指标
        let technical_indicators = calculate_all_indicators(&prices, &highs, &lows, &volumes);
        
        // 3. 计算市场参数
        let returns = calculate_returns(&prices);
        let volatility = calculate_volatility(&returns);
        let trend_strength = calculate_trend_strength(&prices);
        
        // 4. 自适应权重调整
        self.weight_manager.adapt_to_market(volatility, trend_strength);
        
        // 5. 执行金融预测
        let financial_predictions = self.engine.predict(
            &prices,
            &volumes,
            &highs,
            &lows,
            request.prediction_days,
        );
        
        // 6. 转换为标准预测格式
        let predictions = convert_to_standard_predictions(
            &financial_predictions,
            current_price,
            &dates,
            request.prediction_days,
        );
        
        // 7. 风险评估
        let risk_assessment = self.assess_risk(
            &returns,
            volatility,
            current_price,
            &financial_predictions,
        );
        
        // 8. 生成交易建议
        let trading_recommendation = generate_trading_recommendation(
            &financial_predictions,
            &technical_indicators,
            &risk_assessment,
        );
        
        Ok(ProfessionalPredictionResult {
            predictions,
            technical_indicators,
            risk_assessment,
            trading_recommendation,
        })
    }
    
    /// 风险评估
    fn assess_risk(
        &self,
        returns: &[f64],
        volatility: f64,
        current_price: f64,
        predictions: &[FinancialPrediction],
    ) -> RiskAssessment {
        // VaR计算
        let var_95 = self.risk_manager.calculate_var(returns, 0.95, 1);
        
        // 最大回撤
        let prices_with_predictions: Vec<f64> = predictions
            .iter()
            .map(|p| p.predicted_price)
            .collect();
        let max_drawdown = self.risk_manager.calculate_max_drawdown(&prices_with_predictions);
        
        // 夏普比率
        let sharpe_ratio = self.risk_manager.calculate_sharpe_ratio(returns);
        
        // 最优仓位（凯利公式）
        let win_rate = predictions.iter()
            .filter(|p| p.predicted_return > 0.0)
            .count() as f64 / predictions.len() as f64;
        let avg_win = predictions.iter()
            .filter(|p| p.predicted_return > 0.0)
            .map(|p| p.predicted_return)
            .sum::<f64>() / predictions.iter().filter(|p| p.predicted_return > 0.0).count().max(1) as f64;
        let avg_loss = predictions.iter()
            .filter(|p| p.predicted_return < 0.0)
            .map(|p| p.predicted_return.abs())
            .sum::<f64>() / predictions.iter().filter(|p| p.predicted_return < 0.0).count().max(1) as f64;
        
        let win_loss_ratio = if avg_loss > 0.0 { avg_win / avg_loss } else { 1.5 };
        let optimal_position = self.risk_manager.kelly_criterion(win_rate, win_loss_ratio);
        
        // 止损止盈
        let atr = volatility * current_price;
        let stop_loss = self.risk_manager.calculate_stop_loss(current_price, volatility, atr);
        let take_profits = self.risk_manager.calculate_take_profits(
            current_price,
            volatility,
            &[1.5, 2.5, 4.0],
        );
        
        RiskAssessment {
            var_95,
            max_drawdown,
            sharpe_ratio,
            optimal_position,
            stop_loss,
            take_profits,
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

fn calculate_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    
    prices.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

fn calculate_volatility(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.02;
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    variance.sqrt()
}

fn calculate_trend_strength(prices: &[f64]) -> f64 {
    if prices.len() < 20 {
        return 0.5;
    }
    
    // 使用线性回归R²作为趋势强度
    let n = prices.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = prices.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for (i, &y) in prices.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }
    
    if denominator == 0.0 {
        return 0.0;
    }
    
    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;
    
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    
    for (i, &y) in prices.iter().enumerate() {
        let x = i as f64;
        let y_pred = slope * x + intercept;
        ss_res += (y - y_pred).powi(2);
        ss_tot += (y - y_mean).powi(2);
    }
    
    if ss_tot == 0.0 {
        return 0.0;
    }
    
    1.0 - (ss_res / ss_tot)
}

fn convert_to_standard_predictions(
    financial_predictions: &[FinancialPrediction],
    current_price: f64,
    dates: &[String],
    prediction_days: usize,
) -> Vec<Prediction> {
    let mut predictions = Vec::new();
    let last_date = dates.last().cloned().unwrap_or_else(|| "2024-01-01".to_string());
    
    for (i, fp) in financial_predictions.iter().enumerate() {
        let target_date = calculate_future_date(&last_date, i + 1);
        
        predictions.push(Prediction {
            target_date,
            predicted_price: fp.predicted_price,
            predicted_change_percent: fp.predicted_return * 100.0,
            confidence: fp.confidence,
            trading_signal: Some(format!("{:?}", fp.trading_signal.action)),
            signal_strength: Some(fp.trading_signal.strength),
            technical_indicators: None,
            prediction_reason: Some(generate_reason(fp)),
            key_factors: Some(vec![
                format!("市场状态: {:?}", fp.market_regime),
                format!("风险调整收益: {:.2}", fp.risk_adjusted_return),
                format!("波动率: {:.2}%", fp.volatility * 100.0),
            ]),
        });
    }
    
    predictions
}

fn calculate_future_date(base_date: &str, days_ahead: usize) -> String {
    use chrono::{NaiveDate, Duration};
    
    if let Ok(date) = NaiveDate::parse_from_str(base_date, "%Y-%m-%d") {
        let future_date = date + Duration::days(days_ahead as i64);
        future_date.format("%Y-%m-%d").to_string()
    } else {
        format!("T+{}", days_ahead)
    }
}

fn generate_reason(prediction: &FinancialPrediction) -> String {
    let direction = if prediction.predicted_return > 0.0 { "上涨" } else { "下跌" };
    let confidence_level = if prediction.confidence > 0.7 { "高" } else if prediction.confidence > 0.5 { "中" } else { "低" };
    
    format!(
        "基于{:?}市场状态分析，预计{}，置信度{}。风险调整收益率{:.2}，夏普比率{:.2}",
        prediction.market_regime,
        direction,
        confidence_level,
        prediction.risk_adjusted_return,
        prediction.risk_metrics.sharpe_ratio,
    )
}

fn generate_trading_recommendation(
    predictions: &[FinancialPrediction],
    indicators: &TechnicalIndicators,
    risk: &RiskAssessment,
) -> TradingRecommendation {
    if predictions.is_empty() {
        return TradingRecommendation {
            action: "观望".to_string(),
            confidence: 0.5,
            reasons: vec!["数据不足".to_string()],
            risk_level: "未知".to_string(),
        };
    }
    
    let first_pred = &predictions[0];
    let mut reasons = Vec::new();
    
    // 确定交易动作
    let action = match first_pred.trading_signal.action {
        crate::stock_prediction::financial_prediction_engine::TradeAction::StrongBuy => {
            reasons.push("强烈买入信号".to_string());
            "强烈买入"
        }
        crate::stock_prediction::financial_prediction_engine::TradeAction::Buy => {
            reasons.push("买入信号".to_string());
            "买入"
        }
        crate::stock_prediction::financial_prediction_engine::TradeAction::Hold => {
            reasons.push("持有观望".to_string());
            "持有"
        }
        crate::stock_prediction::financial_prediction_engine::TradeAction::Sell => {
            reasons.push("卖出信号".to_string());
            "卖出"
        }
        crate::stock_prediction::financial_prediction_engine::TradeAction::StrongSell => {
            reasons.push("强烈卖出信号".to_string());
            "强烈卖出"
        }
    };
    
    // 添加技术指标原因
    match indicators.rsi.signal {
        crate::stock_prediction::technical_indicators_optimized::RsiSignal::Oversold |
        crate::stock_prediction::technical_indicators_optimized::RsiSignal::ExtremeOversold => {
            reasons.push(format!("RSI超卖({:.1})", indicators.rsi.value));
        }
        crate::stock_prediction::technical_indicators_optimized::RsiSignal::Overbought |
        crate::stock_prediction::technical_indicators_optimized::RsiSignal::ExtemeOverbought => {
            reasons.push(format!("RSI超买({:.1})", indicators.rsi.value));
        }
        _ => {}
    }
    
    // MACD信号
    match indicators.macd.cross_signal {
        crate::stock_prediction::technical_indicators_optimized::CrossSignal::GoldenCross => {
            reasons.push("MACD金叉".to_string());
        }
        crate::stock_prediction::technical_indicators_optimized::CrossSignal::DeathCross => {
            reasons.push("MACD死叉".to_string());
        }
        _ => {}
    }
    
    // 布林带位置
    if indicators.bollinger.position > 0.8 {
        reasons.push("接近布林带上轨".to_string());
    } else if indicators.bollinger.position < 0.2 {
        reasons.push("接近布林带下轨".to_string());
    }
    
    // 风险水平
    let risk_level = if risk.var_95 > 0.05 {
        "高"
    } else if risk.var_95 > 0.02 {
        "中"
    } else {
        "低"
    };
    
    // 添加风险建议
    reasons.push(format!("建议仓位: {:.1}%", risk.optimal_position * 100.0));
    reasons.push(format!("止损位: {:.2}", risk.stop_loss));
    
    TradingRecommendation {
        action: action.to_string(),
        confidence: first_pred.confidence,
        reasons,
        risk_level: risk_level.to_string(),
    }
}

/// 简化的预测入口
pub async fn predict_with_financial_engine(
    request: PredictionRequest,
) -> Result<PredictionResponse, String> {
    let mut predictor = UnifiedPredictor::new();
    let result = predictor.predict(request).await?;
    
    Ok(PredictionResponse {
        predictions: result.predictions,
        last_real_data: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_returns() {
        let prices = vec![100.0, 102.0, 101.0, 103.0];
        let returns = calculate_returns(&prices);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.02).abs() < 0.001);
    }
    
    #[test]
    fn test_trend_strength() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let strength = calculate_trend_strength(&prices);
        assert!(strength > 0.9);  // Strong upward trend
    }
}
