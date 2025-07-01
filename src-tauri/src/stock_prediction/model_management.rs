use std::path::PathBuf;
use std::fs;
use serde_json;
use uuid::Uuid;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::stock_prediction::types::ModelInfo;

// 定义获取模型保存目录的函数
fn get_models_dir() -> PathBuf {
    let app_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("./data"));
    let models_dir = app_dir.join("biga/models");
    fs::create_dir_all(&models_dir).unwrap_or_default();
    models_dir
}

// 定义获取特定模型目录的函数
fn get_model_dir(model_id: &str) -> PathBuf {
    get_models_dir().join(model_id)
}

// 保存模型元数据
pub fn save_model_metadata(metadata: &ModelInfo) -> std::io::Result<()> {
    let model_dir = get_model_dir(&metadata.id);
    fs::create_dir_all(&model_dir)?;
    
    let metadata_path = model_dir.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(metadata)?;
    fs::write(metadata_path, metadata_json)?;
    
    Ok(())
}

// 读取模型元数据
pub fn load_model_metadata(model_id: &str) -> std::io::Result<ModelInfo> {
    let metadata_path = get_model_dir(model_id).join("metadata.json");
    let metadata_json = fs::read_to_string(metadata_path)?;
    let metadata: ModelInfo = serde_json::from_str(&metadata_json)?;
    Ok(metadata)
}

// 列出特定股票代码的所有模型
pub fn list_models(symbol: &str) -> Vec<ModelInfo> {
    let models_dir = get_models_dir();
    
    let mut models = Vec::new();
    
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            
            if path.is_dir() {
                let metadata_path = path.join("metadata.json");
                
                if metadata_path.exists() {
                    if let Ok(metadata_json) = fs::read_to_string(metadata_path) {
                        if let Ok(metadata) = serde_json::from_str::<ModelInfo>(&metadata_json) {
                            if metadata.stock_code == symbol {
                                models.push(metadata);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 按创建时间降序排序
    models.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    models
}

// 删除模型
pub fn delete_model(model_id: &str) -> std::io::Result<()> {
    let model_dir = get_model_dir(model_id);
    fs::remove_dir_all(model_dir)?;
    Ok(())
}

// 生成新的模型ID
pub fn generate_model_id() -> String {
    Uuid::new_v4().to_string()
}

// 获取当前时间戳
pub fn get_current_timestamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

// 获取模型文件路径
pub fn get_model_file_path(model_id: &str) -> PathBuf {
    get_model_dir(model_id).join("model.safetensors")
} 