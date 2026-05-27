//! 截面池化 ML 评测：汇集所有股票样本训练单一 MLP，检验"更多数据+ML"能否击败朴素基准。
//! 每只股票按时间序 80/20 切分（训练=较早、测试=较新），再跨股票拼接。
//! 用法：cargo run --example pooled_ml

use biga_lib::db::connection::create_pool;
use biga_lib::db::repository::get_recent_historical_data;
use biga_lib::prediction::model::features::{build_dataset, FEATURE_DIM};
use biga_lib::prediction::model::network::train_eval;
use sqlx::Row;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    for stmt in [
        "CREATE TABLE IF NOT EXISTS stock_capital (symbol TEXT PRIMARY KEY, circulating_shares REAL NOT NULL DEFAULT 0, total_shares REAL NOT NULL DEFAULT 0, circulating_market_cap REAL NOT NULL DEFAULT 0, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        "ALTER TABLE historical_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
        "ALTER TABLE realtime_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
    ] {
        let _ = sqlx::query(stmt).execute(&pool).await;
    }

    let rows = sqlx::query(
        "SELECT symbol, COUNT(*) c FROM historical_data GROUP BY symbol HAVING c >= 200 ORDER BY c DESC",
    )
    .fetch_all(&pool)
    .await
    .expect("查询股票失败");
    println!("参与池化的股票数：{}", rows.len());

    let mut train_x: Vec<f32> = Vec::new();
    let mut train_y: Vec<f32> = Vec::new();
    let mut test_x: Vec<f32> = Vec::new();
    let mut test_y: Vec<f32> = Vec::new();

    for r in &rows {
        let symbol: String = r.get("symbol");
        // 用较长历史（每股最多 800 条）提供更多样本
        let hist = get_recent_historical_data(&symbol, 800, &pool)
            .await
            .unwrap_or_default();
        let (f, l, n) = build_dataset(&hist);
        if n < 50 {
            continue;
        }
        let n_tr = (n as f64 * 0.8) as usize;
        train_x.extend_from_slice(&f[..n_tr * FEATURE_DIM]);
        train_y.extend_from_slice(&l[..n_tr]);
        test_x.extend_from_slice(&f[n_tr * FEATURE_DIM..]);
        test_y.extend_from_slice(&l[n_tr..]);
    }

    let n_train = train_y.len();
    let n_test = test_y.len();
    println!("池化样本：训练 {n_train}，测试 {n_test}");
    if n_train < 100 || n_test < 50 {
        println!("样本不足，退出");
        return;
    }

    // 多次训练取均值（控随机初始化方差）
    let runs = 3;
    let mut acc_sum = 0.0;
    let mut mae_sum = 0.0;
    let mut ok = 0;
    for _ in 0..runs {
        match train_eval(&train_x, &train_y, n_train, &test_x, &test_y, n_test, 800, 0.005) {
            Ok(o) => {
                acc_sum += o.direction_accuracy;
                mae_sum += o.mae;
                ok += 1;
            }
            Err(e) => println!("训练失败: {e}"),
        }
    }
    if ok == 0 {
        return;
    }
    let ml_acc = acc_sum / ok as f64;

    // 朴素基准：测试集上总猜多数方向
    let up = test_y.iter().filter(|&&v| v > 0.0).count() as f64 / n_test as f64;
    let baseline = up.max(1.0 - up);

    println!(
        "\n>>> 池化 ML 方向准确率 = {:.2}%（{} 次均）",
        ml_acc * 100.0,
        ok
    );
    println!(">>> 朴素基准（总猜多数方向）= {:.2}%", baseline * 100.0);
    println!(">>> 超额 edge = {:+.2}%", (ml_acc - baseline) * 100.0);
    println!(">>> 平均 MAE = {:.3}", mae_sum / ok as f64);
}
