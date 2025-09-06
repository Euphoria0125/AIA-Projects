# NLP模型分析作业

本作业对ResNet18(CNN)和LSTM（RNN）模型进行结构分析、参数量计算和GPU显存占用测量。

## 环境配置

### 硬件要求
- NVIDIA GPU（建议显存≥4GB）
- CUDA兼容显卡（测试使用RTX 3050 + CUDA 11.8）

### 软件依赖
| 组件 | 版本要求 |
|------|----------|
| Python | ≥3.8 |
| PyTorch | ≥2.0 with CUDA |
| torchvision | ≥0.15 |
| matplotlib | ≥3.5 |

### 安装步骤

1. 创建conda环境（推荐）：
```bash
conda create -n nlp_hw python=3.8
conda activate nlp_hw
```

2. 安装依赖库：
```bash
pip install torch torchvision matplotlib pympler
```

3. 验证CUDA可用性：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 项目结构
```
project/
├── model_analysis.py    # 主分析脚本
├── resnet_memory.png    # ResNet显存分析图
├── lstm_memory.png      # LSTM显存分析图
└── README.md           # 本文件
```

## 运行说明

### 基本运行
```bash
python model_analysis.py
```

### 参数调整（可选）
```bash
# 修改ResNet的batch_size测试范围
python model_analysis.py --resnet_batches 1 8 32

# 修改LSTM的sequence_length测试范围
python model_analysis.py --lstm_seqs 64 128 256
```

### 输出文件
运行后将生成：
- `resnet_memory.png`：ResNet显存占用曲线
- `lstm_memory.png`：LSTM显存占用曲线

## 结果解读

### 预期输出示例
```
>>> 开始ResNet18分析 <<<
[显存占用测试]
Batch Size  1: 初始= 45.26MB, 峰值= 75.25MB
Batch Size  8: 初始=143.34MB, 峰值=318.22MB

>>> 开始LSTM分析 <<<
理论参数量: 893,450
实际参数量: 895,498 
差异: 2048 (来自PyTorch实现优化)
```

### 关键指标
1. **ResNet18**：
   - 固定参数：45MB
   - 每样本开销：~30MB

2. **LSTM**：
   - 基础显存：56MB
   - 序列长度影响：每100token增加~20MB

## 常见问题

### Q1: CUDA不可用
```bash
# 重新安装GPU版本PyTorch
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 显存测量为0
- 确认`torch.cuda.is_available()`返回True
- 检查是否误用了CPU版本PyTorch

### Q3: 中文显示警告
```python
# 在代码开头添加：
plt.rcParams['font.sans-serif'] = ['Arial']
```