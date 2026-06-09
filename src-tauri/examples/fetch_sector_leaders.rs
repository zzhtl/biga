//! 按「人工板块」拉取各板块龙头股的全量数据（历史+股本/估值+基本面+量比换手率），
//! 并把板块分类写入 `stock` 表的 category 字段，供股票列表页分组展示。
//!
//! 默认会**先清空全部业务数据**（含已训练模型），再按下方 SECTORS 清单逐只拉取。
//! 单只走统一入口 `refresh_stock_full`，连续 15 次 API 失败自动早停（疑似当日额度耗尽）。
//!
//! 用法：
//!   cargo run --release --example fetch_sector_leaders            # 清库 + 全量拉取
//!   RESUME=1 cargo run --release --example fetch_sector_leaders   # 不清库，跳过已拉到的，断点续拉
//!   SECTOR_MAX=30 cargo run --release --example fetch_sector_leaders  # 限制本次拉取只数（调试）

use biga_lib::db::connection::create_pool;
use biga_lib::services::historical::refresh_stock_full;

/// 各板块龙头清单：(板块名, &[(纯6位代码, 名称)])。约每板块 20 只。
/// 清单为人工策划，可按需增删；拉取时无效代码/历史不足会被跳过。
const SECTORS: &[(&str, &[(&str, &str)])] = &[
    (
        "科技",
        &[
            ("002415", "海康威视"), ("000063", "中兴通讯"), ("002230", "科大讯飞"),
            ("600570", "恒生电子"), ("600588", "用友网络"), ("300454", "深信服"),
            ("688111", "金山办公"), ("600845", "宝信软件"), ("300033", "同花顺"),
            ("002236", "大华股份"), ("300496", "中科创达"), ("600718", "东软集团"),
            ("002241", "歌尔股份"), ("000938", "紫光股份"), ("000977", "浪潮信息"),
            ("603019", "中科曙光"), ("688561", "奇安信"), ("002405", "四维图新"),
            ("300059", "东方财富"), ("600522", "中天科技"),
        ],
    ),
    (
        "能源",
        &[
            ("601857", "中国石油"), ("600028", "中国石化"), ("600938", "中国海油"),
            ("601088", "中国神华"), ("601225", "陕西煤业"), ("601898", "中煤能源"),
            ("600188", "兖矿能源"), ("000983", "山西焦煤"), ("601699", "潞安环能"),
            ("600985", "淮北矿业"), ("000937", "冀中能源"), ("600256", "广汇能源"),
            ("002128", "电投能源"), ("601666", "平煤股份"), ("600740", "山西焦化"),
            ("000968", "蓝焰控股"), ("600971", "恒源煤电"), ("601918", "新集能源"),
            ("000552", "靖远煤电"), ("600395", "盘江股份"),
        ],
    ),
    (
        "矿业",
        &[
            ("601899", "紫金矿业"), ("603993", "洛阳钼业"), ("600547", "山东黄金"),
            ("601600", "中国铝业"), ("000831", "中国稀土"), ("600111", "北方稀土"),
            ("000060", "中金岭南"), ("000878", "云南铜业"), ("000630", "铜陵有色"),
            ("601168", "西部矿业"), ("600362", "江西铜业"), ("600489", "中金黄金"),
            ("002155", "湖南黄金"), ("601020", "华钰矿业"), ("000975", "银泰黄金"),
            ("000688", "国城矿业"), ("600331", "宏达股份"), ("601958", "金钼股份"),
            ("000758", "中色股份"), ("600459", "贵研铂业"),
        ],
    ),
    (
        "电力",
        &[
            ("600900", "长江电力"), ("601985", "中国核电"), ("600886", "国投电力"),
            ("601991", "大唐发电"), ("600025", "华能水电"), ("600011", "华能国际"),
            ("003816", "中国广核"), ("600795", "国电电力"), ("600027", "华电国际"),
            ("000539", "粤电力A"), ("600021", "上海电力"), ("601868", "中国能建"),
            ("600236", "桂冠电力"), ("000883", "湖北能源"), ("000027", "深圳能源"),
            ("600674", "川投能源"), ("000591", "太阳能"), ("600905", "三峡能源"),
            ("601619", "嘉泽新能"), ("000875", "吉电股份"),
        ],
    ),
    (
        "能源金属",
        &[
            ("002460", "赣锋锂业"), ("002466", "天齐锂业"), ("300618", "寒锐钴业"),
            ("603799", "华友钴业"), ("002176", "江特电机"), ("002756", "永兴材料"),
            ("002738", "中矿资源"), ("000762", "西藏矿业"), ("603026", "胜华新材"),
            ("688779", "长远锂科"), ("300769", "德方纳米"), ("002709", "天赐材料"),
            ("300390", "天华新能"), ("688567", "孚能科技"), ("300035", "中科电气"),
            ("002245", "蔚蓝锂芯"), ("688116", "天奈科技"), ("002074", "国轩高科"),
            ("300073", "当升科技"), ("688005", "容百科技"),
        ],
    ),
    (
        "消费",
        &[
            ("600519", "贵州茅台"), ("000858", "五粮液"), ("000568", "泸州老窖"),
            ("600809", "山西汾酒"), ("000596", "古井贡酒"), ("002304", "洋河股份"),
            ("600600", "青岛啤酒"), ("600887", "伊利股份"), ("000895", "双汇发展"),
            ("603288", "海天味业"), ("000333", "美的集团"), ("000651", "格力电器"),
            ("600690", "海尔智家"), ("603899", "晨光股份"), ("002507", "涪陵榨菜"),
            ("603517", "绝味食品"), ("000799", "酒鬼酒"), ("600132", "重庆啤酒"),
            ("603866", "桃李面包"), ("601888", "中国中免"),
        ],
    ),
    (
        "半导体",
        &[
            ("688981", "中芯国际"), ("603501", "韦尔股份"), ("002371", "北方华创"),
            ("688012", "中微公司"), ("603986", "兆易创新"), ("688041", "海光信息"),
            ("688008", "澜起科技"), ("603160", "汇顶科技"), ("688126", "沪硅产业"),
            ("688396", "华润微"), ("002049", "紫光国微"), ("600460", "士兰微"),
            ("688256", "寒武纪"), ("688082", "盛美上海"), ("603290", "斯达半导"),
            ("688516", "奥特维"), ("688368", "晶丰明源"), ("688728", "格科微"),
            ("300782", "卓胜微"), ("002916", "深南电路"),
        ],
    ),
    (
        "军工汽车",
        &[
            ("600760", "中航沈飞"), ("000768", "中航西飞"), ("600893", "航发动力"),
            ("600150", "中国船舶"), ("002013", "中航机载"), ("600316", "洪都航空"),
            ("000547", "航天发展"), ("600118", "中国卫星"), ("002179", "中航光电"),
            ("600038", "中直股份"), ("002594", "比亚迪"), ("600104", "上汽集团"),
            ("601633", "长城汽车"), ("000625", "长安汽车"), ("601238", "广汽集团"),
            ("000338", "潍柴动力"), ("601127", "赛力斯"), ("600660", "福耀玻璃"),
            ("002050", "三花智控"), ("002920", "德赛西威"),
        ],
    ),
    (
        "医药生物",
        &[
            ("600276", "恒瑞医药"), ("300760", "迈瑞医疗"), ("603259", "药明康德"),
            ("300347", "泰格医药"), ("000538", "云南白药"), ("600196", "复星医药"),
            ("002422", "科伦药业"), ("600436", "片仔癀"), ("300122", "智飞生物"),
            ("603392", "万泰生物"), ("000661", "长春高新"), ("002821", "凯莱英"),
            ("300759", "康龙化成"), ("600867", "通化东宝"), ("688180", "君实生物"),
            ("300601", "康泰生物"), ("002007", "华兰生物"), ("600085", "同仁堂"),
            ("000963", "华东医药"), ("300015", "爱尔眼科"),
        ],
    ),
];

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");
    let resume = std::env::var("RESUME").map(|v| v == "1").unwrap_or(false);
    let max = std::env::var("SECTOR_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    // 每只之间的间隔（毫秒）。token 触发 429 限流时可调大（如 1500）续拉。
    let req_delay_ms = std::env::var("REQ_DELAY_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(300);

    // 确保 stock 表存在 category 列（迁移在 App 启动时跑，独立 example 需自行幂等补列）。
    let _ = sqlx::query("ALTER TABLE stock ADD COLUMN category TEXT NOT NULL DEFAULT ''")
        .execute(&pool)
        .await; // duplicate column 报错忽略

    if resume {
        println!("RESUME=1：跳过清库，仅补拉未完成的票。");
    } else {
        println!("清空全部业务数据（含已训练模型）……");
        for table in [
            "historical_data", "realtime_data", "stock_capital", "stock_fundamentals",
            "stock_info", "stock", "stock_predictions", "stock_prediction_models",
        ] {
            match sqlx::query(&format!("DELETE FROM {table}")).execute(&pool).await {
                Ok(r) => println!("  清空 {table}: {} 行", r.rows_affected()),
                Err(e) => println!("  清空 {table} 跳过: {e}"),
            }
        }
    }

    let total: usize = SECTORS.iter().map(|(_, v)| v.len()).sum();
    println!("板块 {} 个，候选龙头共 {total} 只。\n", SECTORS.len());

    let (mut ok, mut skipped, mut failed, mut consec_fail) = (0u32, 0u32, 0u32, 0u32);
    let mut done = 0usize;
    'outer: for (category, members) in SECTORS {
        println!("==== 板块「{category}」({} 只) ====", members.len());
        for (code, name) in *members {
            if done >= max {
                println!("已达 SECTOR_MAX={max}，停止。");
                break 'outer;
            }
            done += 1;
            let exchange = if code.starts_with('6') { "sh" } else { "sz" };

            // 续拉模式：已有足够历史则只确保分类落库，跳过拉取。
            if resume {
                let bars: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM historical_data WHERE symbol = ?")
                    .bind(*code)
                    .fetch_one(&pool)
                    .await
                    .unwrap_or(0);
                if bars >= 120 {
                    upsert_category(&pool, code, name, category, exchange).await;
                    skipped += 1;
                    continue;
                }
            }

            // 必须先写 stock_info：refresh 内部插入 realtime_data 时要按代码取名字。
            ensure_stock_info(&pool, code, name, exchange).await;

            match refresh_stock_full(code, &pool).await {
                Ok(summary) => {
                    consec_fail = 0;
                    upsert_category(&pool, code, name, category, exchange).await;
                    ok += 1;
                    println!(
                        "  [{done}/{total}] {code} {name}: 历史{}条, 股本{}, 基本面{}期 ✓",
                        summary.bars,
                        if summary.capital_updated { "✓" } else { "—" },
                        summary.fundamental_reports
                    );
                }
                Err(e) => {
                    failed += 1;
                    consec_fail += 1;
                    println!("  [{done}/{total}] {code} {name}: 拉取失败（连续 {consec_fail}）: {e}");
                    if consec_fail >= 15 {
                        println!("⚠️ 连续 {consec_fail} 次失败，疑似当日额度耗尽，提前停止。");
                        println!("   额度恢复后用 RESUME=1 续拉。");
                        break 'outer;
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(req_delay_ms)).await;
        }
    }

    println!(
        "\n完成：成功 {ok} / 跳过 {skipped} / 失败 {failed}（共处理 {done} 只）。"
    );
}

/// 写入 stock_info（refresh 内部插入 realtime_data 取名字依赖它，故须在拉取前调用）。
async fn ensure_stock_info(pool: &sqlx::SqlitePool, code: &str, name: &str, exchange: &str) {
    let _ = sqlx::query(
        "INSERT INTO stock_info (symbol, name, exchange) VALUES (?, ?, ?)
         ON CONFLICT(symbol) DO UPDATE SET name = excluded.name, exchange = excluded.exchange",
    )
    .bind(code)
    .bind(name)
    .bind(exchange)
    .execute(pool)
    .await;
}

/// 把名称+板块分类落入 stock 表（股票列表页按 category 分组的数据源），拉取成功后调用。
async fn upsert_category(pool: &sqlx::SqlitePool, code: &str, name: &str, category: &str, exchange: &str) {
    let _ = sqlx::query(
        "INSERT INTO stock (symbol, name, exchange, category) VALUES (?, ?, ?, ?)
         ON CONFLICT(symbol) DO UPDATE SET name = excluded.name, category = excluded.category, exchange = excluded.exchange",
    )
    .bind(code)
    .bind(name)
    .bind(exchange)
    .bind(category)
    .execute(pool)
    .await;
}
