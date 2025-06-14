use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Linear, Module, VarBuilder, VarMap, AdamW, Embedding};
use candle_nn::ops::dropout;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

// MLP模型定义
pub struct MLP {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
    dropout: f64,
}

unsafe impl Send for MLP {}
unsafe impl Sync for MLP {}

impl MLP {
    pub fn new(
        in_dim: usize, 
        hidden_dim: usize, 
        out_dim: usize, 
        dropout: f64, 
        vb: VarBuilder
    ) -> Result<Self> {
        let ln1 = candle_nn::linear(in_dim, hidden_dim, vb.pp("ln1"))?;
        let ln2 = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("ln2"))?;
        let ln3 = candle_nn::linear(hidden_dim, out_dim, vb.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3, dropout })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = if train { dropout(&xs, self.dropout as f32)? } else { xs.clone() };
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = if train { dropout(&xs, self.dropout as f32)? } else { xs.clone() };
        let xs = self.ln3.forward(&xs)?;
        Ok(xs)
    }
}

// LSTM单元
struct LSTMCell {
    wx: Linear,
    wh: Linear,
    hidden_size: usize,
}

unsafe impl Send for LSTMCell {}
unsafe impl Sync for LSTMCell {}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let wx = candle_nn::linear(input_size, 4 * hidden_size, vb.pp("wx"))?;
        let wh = candle_nn::linear(hidden_size, 4 * hidden_size, vb.pp("wh"))?;
        Ok(Self { wx, wh, hidden_size })
    }

    fn forward(&self, x: &Tensor, h_prev: &Tensor, c_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let wx = self.wx.forward(x)?;
        let wh = self.wh.forward(h_prev)?;
        let gates = wx.add(&wh)?;
        
        // 使用自定义sigmoid实现替代内置方法
        let i_gate = gates.narrow(1, 0, self.hidden_size)?.sigmoid_custom()?;
        let f_gate = gates.narrow(1, self.hidden_size, self.hidden_size)?.sigmoid_custom()?;
        let g_gate = gates.narrow(1, 2 * self.hidden_size, self.hidden_size)?.tanh()?;
        let o_gate = gates.narrow(1, 3 * self.hidden_size, self.hidden_size)?.sigmoid_custom()?;
        
        let c = f_gate.mul(c_prev)?.add(&i_gate.mul(&g_gate)?)?;
        let h = o_gate.mul(&c.tanh()?)?;
        
        Ok((h, c))
    }
}

// LSTM模型
pub struct LSTM {
    cell: LSTMCell,
    hidden_size: usize,
    output_layer: Linear,
    dropout: f64,
}

unsafe impl Send for LSTM {}
unsafe impl Sync for LSTM {}

impl LSTM {
    pub fn new(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize, 
        dropout: f64,
        vb: VarBuilder
    ) -> Result<Self> {
        let cell = LSTMCell::new(input_size, hidden_size, vb.pp("cell"))?;
        let output_layer = candle_nn::linear(hidden_size, output_size, vb.pp("output"))?;
        Ok(Self { cell, hidden_size, output_layer, dropout })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (batch_size, seq_len, _) = xs.dims3()?;
        let device = xs.device();
        
        let mut h = Tensor::zeros(&[batch_size, self.hidden_size], xs.dtype(), device)?;
        let mut c = Tensor::zeros(&[batch_size, self.hidden_size], xs.dtype(), device)?;
        
        let mut outputs = Vec::with_capacity(seq_len);
        
        for i in 0..seq_len {
            let x = xs.narrow(1, i, 1)?.squeeze(1)?;
            let (new_h, new_c) = self.cell.forward(&x, &h, &c)?;
            h = if train { dropout(&new_h, self.dropout as f32)? } else { new_h };
            c = new_c;
            outputs.push(h.clone());
        }
        
        let last_output = outputs.last().unwrap();
        let predictions = self.output_layer.forward(last_output)?;
        Ok(predictions)
    }
}

// GRU单元
struct GRUCell {
    w_z: Linear, // 更新门权重
    w_r: Linear, // 重置门权重
    w_h: Linear, // 输出激活权重
}

unsafe impl Send for GRUCell {}
unsafe impl Sync for GRUCell {}

impl GRUCell {
    fn new(input_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let w_z = candle_nn::linear(input_size + hidden_size, hidden_size, vb.pp("w_z"))?;
        let w_r = candle_nn::linear(input_size + hidden_size, hidden_size, vb.pp("w_r"))?;
        let w_h = candle_nn::linear(input_size + hidden_size, hidden_size, vb.pp("w_h"))?;
        Ok(Self { w_z, w_r, w_h })
    }

    fn forward(&self, x: &Tensor, h_prev: &Tensor) -> Result<Tensor> {
        // 组合输入和前一个隐藏状态
        let x_h = Tensor::cat(&[x, h_prev], 1)?;
        
        // 计算更新门和重置门
        let z = self.w_z.forward(&x_h)?.sigmoid_custom()?;
        let r = self.w_r.forward(&x_h)?.sigmoid_custom()?;
        
        // 计算候选隐藏状态
        let r_h = r.mul(h_prev)?;
        let x_rh = Tensor::cat(&[x, &r_h], 1)?;
        let h_tilde = self.w_h.forward(&x_rh)?.tanh()?;
        
        // 更新隐藏状态
        let z_h = z.mul(h_prev)?;
        let one_minus_z = Tensor::ones_like(&z)?.sub(&z)?;
        let one_minus_z_h = one_minus_z.mul(&h_tilde)?;
        let h = z_h.add(&one_minus_z_h)?;
        
        Ok(h)
    }
}

// GRU模型
pub struct GRU {
    cell: GRUCell,
    hidden_size: usize,
    output_layer: Linear,
    dropout: f64,
}

unsafe impl Send for GRU {}
unsafe impl Sync for GRU {}

impl GRU {
    pub fn new(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize, 
        dropout: f64,
        vb: VarBuilder
    ) -> Result<Self> {
        let cell = GRUCell::new(input_size, hidden_size, vb.pp("cell"))?;
        let output_layer = candle_nn::linear(hidden_size, output_size, vb.pp("output"))?;
        Ok(Self { cell, hidden_size, output_layer, dropout })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (batch_size, seq_len, _) = xs.dims3()?;
        let device = xs.device();
        
        let mut h = Tensor::zeros(&[batch_size, self.hidden_size], xs.dtype(), device)?;
        
        for i in 0..seq_len {
            let x = xs.narrow(1, i, 1)?.squeeze(1)?;
            let new_h = self.cell.forward(&x, &h)?;
            h = if train { dropout(&new_h, self.dropout as f32)? } else { new_h };
        }
        
        let predictions = self.output_layer.forward(&h)?;
        Ok(predictions)
    }
}

// 自注意力层
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    head_dim: usize,
    n_heads: usize,
}

unsafe impl Send for SelfAttention {}
unsafe impl Sync for SelfAttention {}

impl SelfAttention {
    fn new(hidden_dim: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_dim / n_heads;
        let query = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("query"))?;
        let key = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("key"))?;
        let value = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("value"))?;
        let output = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("output"))?;
        
        Ok(Self { query, key, value, output, head_dim, n_heads })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = xs.dims3()?;
        
        // 线性变换
        let queries = self.query.forward(xs)?;
        let keys = self.key.forward(xs)?;
        let values = self.value.forward(xs)?;
        
        // 调整形状为多头
        let queries = queries.reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
                            .transpose(1, 2)?; // [batch, heads, seq, head_dim]
        let keys = keys.reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
                      .transpose(1, 2)?; // [batch, heads, seq, head_dim]
        let values = values.reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
                          .transpose(1, 2)?; // [batch, heads, seq, head_dim]
        
        // 计算注意力分数
        let scores = queries.matmul(&keys.transpose(2, 3)?)?; // [batch, heads, seq, seq]
        let scale = (self.head_dim as f64).sqrt();
        // 创建包含scale值的Tensor
        let scale_tensor = Tensor::new(&[scale], scores.device())?;
        let scaled_scores = scores.div(&scale_tensor)?;
        
        // 在第3个维度上应用softmax（替代softmax方法）
        let attention = scaled_scores.exp()?;
        let sum = attention.sum_keepdim(3)?;
        let attention = attention.div(&sum)?;
        
        // 计算加权和
        let context = attention.matmul(&values)?; // [batch, heads, seq, head_dim]
        
        // 恢复原始形状
        let context = context.transpose(1, 2)? // [batch, seq, heads, head_dim]
                           .reshape(&[batch_size, seq_len, self.n_heads * self.head_dim])?;
        
        // 输出投影
        let output = self.output.forward(&context)?;
        
        Ok(output)
    }
}

// 前馈网络层
struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    dropout: f64,
}

unsafe impl Send for FeedForward {}
unsafe impl Sync for FeedForward {}

impl FeedForward {
    fn new(hidden_dim: usize, ff_dim: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(hidden_dim, ff_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(ff_dim, hidden_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2, dropout })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        let xs = if train { dropout(&xs, self.dropout as f32)? } else { xs.clone() };
        let xs = self.fc2.forward(&xs)?;
        Ok(xs)
    }
}

// Transformer编码层
struct TransformerEncoderLayer {
    attention: SelfAttention,
    norm1: candle_nn::LayerNorm,
    feed_forward: FeedForward,
    norm2: candle_nn::LayerNorm,
    dropout: f64,
}

unsafe impl Send for TransformerEncoderLayer {}
unsafe impl Sync for TransformerEncoderLayer {}

impl TransformerEncoderLayer {
    fn new(hidden_dim: usize, n_heads: usize, ff_dim: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let attention = SelfAttention::new(hidden_dim, n_heads, vb.pp("attention"))?;
        let norm1 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("norm1"))?;
        let feed_forward = FeedForward::new(hidden_dim, ff_dim, dropout, vb.pp("feed_forward"))?;
        let norm2 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("norm2"))?;
        Ok(Self { attention, norm1, feed_forward, norm2, dropout })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // 自注意力 + 残差
        let attn_out = self.attention.forward(xs)?;
        let attn_out = if train { dropout(&attn_out, self.dropout as f32)? } else { attn_out };
        let xs = xs.add(&attn_out)?;
        
        // 第一个层归一化
        let xs = self.norm1.forward(&xs)?;
        
        // 前馈网络 + 残差
        let ff_out = self.feed_forward.forward(&xs, train)?;
        let ff_out = if train { dropout(&ff_out, self.dropout as f32)? } else { ff_out };
        let xs = xs.add(&ff_out)?;
        
        // 第二个层归一化
        let xs = self.norm2.forward(&xs)?;
        
        Ok(xs)
    }
}

// Transformer模型
pub struct Transformer {
    input_embedding: Linear,
    position_embedding: Embedding,
    encoder_layers: Vec<TransformerEncoderLayer>,
    output_layer: Linear,
    seq_len: usize,
    dropout: f64,
}

unsafe impl Send for Transformer {}
unsafe impl Sync for Transformer {}

impl Transformer {
    pub fn new(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize,
        n_heads: usize,
        n_layers: usize, 
        max_seq_len: usize,
        dropout: f64,
        vb: VarBuilder
    ) -> Result<Self> {
        let input_embedding = candle_nn::linear(input_size, hidden_size, vb.pp("input_emb"))?;
        let position_embedding = candle_nn::embedding(max_seq_len, hidden_size, vb.pp("pos_emb"))?;
        
        let mut encoder_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let layer = TransformerEncoderLayer::new(
                hidden_size, 
                n_heads, 
                hidden_size * 4, // 前馈网络维度通常是隐藏层的4倍
                dropout,
                vb.pp(&format!("layer{i}"))
            )?;
            encoder_layers.push(layer);
        }
        
        let output_layer = candle_nn::linear(hidden_size, output_size, vb.pp("output"))?;
        
        Ok(Self {
            input_embedding,
            position_embedding,
            encoder_layers,
            output_layer,
            seq_len: max_seq_len,
            dropout,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (batch_size, seq_len, _) = xs.dims3()?;
        
        // 输入嵌入
        let xs = self.input_embedding.forward(xs)?;
        
        // 位置编码
        let positions = Tensor::arange(0, seq_len as u32, xs.device())?
            .unsqueeze(0)?
            .expand(&[batch_size, seq_len])?;
        let pos_emb = self.position_embedding.forward(&positions)?;
        
        // 添加位置编码
        let mut xs = xs.add(&pos_emb)?;
        xs = if train { dropout(&xs, self.dropout as f32)? } else { xs };
        
        // 通过编码层
        for layer in &self.encoder_layers {
            xs = layer.forward(&xs, train)?;
        }
        
        // 获取序列的最后一个向量
        let last_hidden = xs.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        
        // 输出层
        let output = self.output_layer.forward(&last_hidden)?;
        
        Ok(output)
    }
}

// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub dropout: f64,
    pub learning_rate: f64,
    #[serde(default = "default_layers")]
    pub n_layers: usize,
    #[serde(default = "default_heads")]
    pub n_heads: usize,
    #[serde(default = "default_seq_len")]
    pub max_seq_len: usize,
}

fn default_layers() -> usize {
    2
}

fn default_heads() -> usize {
    4
}

fn default_seq_len() -> usize {
    60
}

// 自定义sigmoid实现
trait SigmoidExt {
    fn sigmoid_custom(&self) -> Result<Tensor>;
}

impl SigmoidExt for Tensor {
    fn sigmoid_custom(&self) -> Result<Tensor> {
        // sigmoid = 1 / (1 + exp(-x))
        let neg_x = self.neg()?;
        let exp_neg_x = neg_x.exp()?;
        // 创建与self形状相同的全1张量
        let one = Tensor::ones_like(&exp_neg_x)?;
        let one_plus_exp = exp_neg_x.add(&one)?;
        let result = one_plus_exp.recip()?;
        Ok(result)
    }
}

// 保存模型
pub fn save_model(varmap: &VarMap, path: &Path) -> Result<()> {
    // 创建临时文件并将其路径传递给 save 方法
    varmap.save(path)?;
    Ok(())
}

// 创建模型
pub fn create_model(
    config: &ModelConfig, 
    device: &Device
) -> Result<(VarMap, Box<dyn Module + Send + Sync>)> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    let model: Box<dyn Module + Send + Sync> = match config.model_type.as_str() {
        "candle_mlp" => {
            let mlp = MLP::new(
                config.input_size, 
                config.hidden_size, 
                config.output_size,
                config.dropout, 
                vs
            )?;
            Box::new(MLPModule::new(mlp))
        },
        "candle_lstm" => {
            let lstm = LSTM::new(
                config.input_size,
                config.hidden_size,
                config.output_size,
                config.dropout,
                vs
            )?;
            Box::new(LSTMModule::new(lstm))
        },
        "candle_gru" => {
            let gru = GRU::new(
                config.input_size,
                config.hidden_size,
                config.output_size,
                config.dropout,
                vs
            )?;
            Box::new(GRUModule::new(gru))
        },
        "candle_transformer" => {
            let transformer = Transformer::new(
                config.input_size,
                config.hidden_size,
                config.output_size,
                config.n_heads,
                config.n_layers,
                config.max_seq_len,
                config.dropout,
                vs
            )?;
            Box::new(TransformerModule::new(transformer))
        },
        _ => return Err(candle_core::Error::Msg(format!("未支持的模型类型: {}", config.model_type))),
    };
    
    Ok((varmap, model))
}

// 包装结构体，用于实现Module特性
struct MLPModule {
    inner: MLP,
}

unsafe impl Send for MLPModule {}
unsafe impl Sync for MLPModule {}

impl MLPModule {
    fn new(inner: MLP) -> Self {
        Self { inner }
    }
}

impl Module for MLPModule {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs, false)
    }
}

struct LSTMModule {
    inner: LSTM,
}

unsafe impl Send for LSTMModule {}
unsafe impl Sync for LSTMModule {}

impl LSTMModule {
    fn new(inner: LSTM) -> Self {
        Self { inner }
    }
}

impl Module for LSTMModule {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs, false)
    }
}

struct GRUModule {
    inner: GRU,
}

unsafe impl Send for GRUModule {}
unsafe impl Sync for GRUModule {}

impl GRUModule {
    fn new(inner: GRU) -> Self {
        Self { inner }
    }
}

impl Module for GRUModule {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs, false)
    }
}

struct TransformerModule {
    inner: Transformer,
}

unsafe impl Send for TransformerModule {}
unsafe impl Sync for TransformerModule {}

impl TransformerModule {
    fn new(inner: Transformer) -> Self {
        Self { inner }
    }
}

impl Module for TransformerModule {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs, false)
    }
}
