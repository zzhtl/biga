//! 背离建议动作生成

/// 生成背离建议动作
#[allow(dead_code)]
pub(super) fn generate_divergence_action(score: f64, count: usize, confidence: f64) -> String {
    if count == 0 {
        return "无明显背离信号，维持当前策略".to_string();
    }

    let strength = if count >= 3 {
        "强"
    } else if count >= 2 {
        "中等"
    } else {
        "弱"
    };

    if score > 0.5 && confidence > 0.65 {
        format!("{}多重底背离信号，考虑逢低布局", strength)
    } else if score > 0.3 && confidence > 0.55 {
        format!("{}底背离信号，关注反弹机会", strength)
    } else if score < -0.5 && confidence > 0.65 {
        format!("{}多重顶背离信号，考虑减仓或对冲", strength)
    } else if score < -0.3 && confidence > 0.55 {
        format!("{}顶背离信号，警惕回调风险", strength)
    } else {
        "背离信号强度不足，建议观望".to_string()
    }
}

/// 增强版背离建议
pub(super) fn generate_divergence_action_enhanced(
    score: f64,
    count: usize,
    confidence: f64,
    is_triple: bool,
    hidden_count: usize,
) -> String {
    if count == 0 {
        return "无明显背离信号，维持当前策略".to_string();
    }

    // 三重背离特殊处理
    if is_triple {
        if score > 0.3 {
            return format!(
                "⚠️ 三重底背离！极强反转信号，置信度{:.0}%，建议积极布局",
                confidence * 100.0
            );
        } else if score < -0.3 {
            return format!(
                "⚠️ 三重顶背离！极强见顶信号，置信度{:.0}%，建议立即减仓",
                confidence * 100.0
            );
        }
    }

    let strength = if count >= 4 {
        "极强"
    } else if count >= 3 {
        "强"
    } else if count >= 2 {
        "中等"
    } else {
        "弱"
    };

    // 隐藏背离提示
    let hidden_note = if hidden_count > 0 {
        format!("(含{}个隐藏背离，趋势可能延续)", hidden_count)
    } else {
        String::new()
    };

    if score > 0.5 && confidence > 0.65 {
        format!("{}多重底背离信号{}，考虑逢低布局", strength, hidden_note)
    } else if score > 0.3 && confidence > 0.55 {
        format!("{}底背离信号{}，关注反弹机会", strength, hidden_note)
    } else if score < -0.5 && confidence > 0.65 {
        format!("{}多重顶背离信号{}，考虑减仓或对冲", strength, hidden_note)
    } else if score < -0.3 && confidence > 0.55 {
        format!("{}顶背离信号{}，警惕回调风险", strength, hidden_note)
    } else {
        "背离信号强度不足，建议观望".to_string()
    }
}
