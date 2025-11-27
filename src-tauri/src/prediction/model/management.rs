//! 模型管理模块

use crate::prediction::types::ModelInfo;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

/// 获取模型存储目录
pub fn get_models_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let models_dir = home.join(".biga").join("models");
    
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).ok();
    }
    
    models_dir
}

/// 获取模型文件路径
pub fn get_model_file_path(model_id: &str) -> PathBuf {
    get_models_dir().join(format!("{model_id}.safetensors"))
}

/// 获取模型元数据路径
pub fn get_metadata_file_path(model_id: &str) -> PathBuf {
    get_models_dir().join(format!("{model_id}.json"))
}

/// 生成模型 ID
pub fn generate_model_id() -> String {
    Uuid::new_v4().to_string()
}

/// 获取当前时间戳
pub fn get_current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 保存模型元数据
pub fn save_model_metadata(metadata: &ModelInfo) -> Result<(), String> {
    let path = get_metadata_file_path(&metadata.id);
    let json = serde_json::to_string_pretty(metadata)
        .map_err(|e| format!("序列化元数据失败: {e}"))?;
    
    fs::write(&path, json)
        .map_err(|e| format!("写入元数据失败: {e}"))?;
    
    Ok(())
}

/// 加载模型元数据
pub fn load_model_metadata(model_id: &str) -> Result<ModelInfo, String> {
    let path = get_metadata_file_path(model_id);
    let json = fs::read_to_string(&path)
        .map_err(|e| format!("读取元数据失败: {e}"))?;
    
    serde_json::from_str(&json)
        .map_err(|e| format!("解析元数据失败: {e}"))
}

/// 列出指定股票的所有模型
pub fn list_models(stock_code: &str) -> Vec<ModelInfo> {
    let models_dir = get_models_dir();
    let mut models = Vec::new();
    
    if let Ok(entries) = fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(json) = fs::read_to_string(&path) {
                    if let Ok(metadata) = serde_json::from_str::<ModelInfo>(&json) {
                        if metadata.stock_code == stock_code {
                            models.push(metadata);
                        }
                    }
                }
            }
        }
    }
    
    // 按创建时间倒序排列
    models.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    models
}

/// 删除模型
pub fn delete_model(model_id: &str) -> Result<(), String> {
    let model_path = get_model_file_path(model_id);
    let metadata_path = get_metadata_file_path(model_id);
    
    if model_path.exists() {
        fs::remove_file(&model_path)
            .map_err(|e| format!("删除模型文件失败: {e}"))?;
    }
    
    if metadata_path.exists() {
        fs::remove_file(&metadata_path)
            .map_err(|e| format!("删除元数据文件失败: {e}"))?;
    }
    
    Ok(())
}

/// 检查模型是否存在
pub fn model_exists(model_id: &str) -> bool {
    get_model_file_path(model_id).exists()
}

/// 获取模型大小（字节）
pub fn get_model_size(model_id: &str) -> Option<u64> {
    let path = get_model_file_path(model_id);
    fs::metadata(&path).ok().map(|m| m.len())
}

