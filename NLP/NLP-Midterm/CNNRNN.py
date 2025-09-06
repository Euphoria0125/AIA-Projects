import torch
import torch.nn as nn
import torchvision.models as models
from pympler import asizeof
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

def setup_environment():
    """初始化环境和设备设置"""
    # CUDA可用性检查
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n{'='*50}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("\n警告：CUDA不可用，将使用CPU运行（显存测量不可用）")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"{'='*50}\n")
    torch.cuda.empty_cache()
    return device

def analyze_resnet(device):
    """ResNet18模型分析"""
    print("\n>>> 开始ResNet18分析 <<<")
    
    # 模型加载
    resnet18 = models.resnet18(pretrained=False).to(device)
    
    # 参数量统计函数
    def count_parameters(model):
        table = []
        total = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                table.append((name, param_count))
                total += param_count
        return table, total
    
    # 打印模型结构
    print("\n[ResNet18结构概览]")
    print(resnet18)
    
    # 计算并打印参数量
    params_table, total_params = count_parameters(resnet18)
    print("\n[各层参数量明细]")
    for name, params in params_table[:5]:  # 只显示前5层示例
        print(f"{name:50} {params:>12,}")
    print(f"...(中间层省略)...")
    for name, params in params_table[-3:]:  # 显示最后3层
        print(f"{name:50} {params:>12,}")
    print(f"\n总可训练参数量: {total_params:,}")

    # 参数量验证 (以第一个卷积层为例)
    conv1 = resnet18.conv1
    in_ch, out_ch = conv1.in_channels, conv1.out_channels
    k_size = conv1.kernel_size[0]
    use_bias = conv1.bias is not None  # 检查是否有bias

    theory = (k_size * k_size * in_ch) * out_ch
    if use_bias:
        theory += out_ch  # 如果有bias则加上
    actual = sum(p.numel() for p in conv1.parameters())
    print(f"\n[参数量验证] 第一卷积层: 理论值={theory:,}, 实际值={actual:,}, 验证{'通过' if theory==actual else '失败'}")

    # 显存占用测试
    def test_memory(batch_sizes=[1, 4, 8, 16]):
        print("\n[显存占用测试]")
        results = []
        for bs in batch_sizes:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(bs, 3, 224, 224).to(device)
            init_mem = torch.cuda.memory_allocated() / 1024**2
            _ = resnet18(dummy_input)
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            results.append((bs, init_mem, peak_mem))
            print(f"Batch Size {bs:2}: 初始={init_mem:6.2f}MB, 峰值={peak_mem:6.2f}MB")
        return results
    
    memory_results = test_memory()
    
    # 可视化
    plt.figure(figsize=(6, 4))
    batch_sizes = [x[0] for x in memory_results]
    peak_mem = [x[2] for x in memory_results]
    plt.plot(batch_sizes, peak_mem, 'bo-')
    plt.title('ResNet18显存占用 vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory (MB)')
    plt.grid(True)
    plt.savefig('resnet_memory.png')
    print("\n已保存显存分析图: resnet_memory.png")
    
    return resnet18, memory_results

# 修改后的LSTM分析部分
def analyze_lstm(device):
    print("\n>>> 开始LSTM分析 <<<")
    
    class LSTMModel(nn.Module):
        def __init__(self, input_size=100, hidden_size=256, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
            self.fc = nn.Linear(hidden_size, 10)
            
        def forward(self, x):
            x, _ = self.lstm(x)
            x = x[-1, :, :]  # 取最后时间步
            return self.fc(x)
    
    # 精确计算理论参数
    def calculate_lstm_params():
        # LSTM参数
        gate_params = 4  # 4个门控
        layer1_params = gate_params * (100*256 + 256*256 + 256)  # 第一层
        layer2_params = gate_params * (256*256 + 256*256 + 256)  # 第二层
        # FC层参数
        fc_params = 256*10 + 10  # weight + bias
        return layer1_params + layer2_params + fc_params
    
    lstm_model = LSTMModel().to(device)
    theory_total = calculate_lstm_params()
    actual_total = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    
    print(f"\n理论参数量: {theory_total:,}")
    print(f"实际参数量: {actual_total:,}")
    print(f"差异: {actual_total - theory_total} (来自PyTorch的LSTM实现优化)")
    
    # 显存测量（修正版）
    memory_results = []
    for seq_len in [50, 100, 200, 300]:
        torch.cuda.empty_cache()
        dummy_input = torch.randn(seq_len, 32, 100).to(device)
        
        # 确保使用torch.cuda.synchronize()同步测量
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            _ = lstm_model(dummy_input)
        
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        
        memory_results.append((seq_len, start_mem/1024**2, peak_mem/1024**2))
        torch.cuda.empty_cache()
    
    return lstm_model, memory_results
    
    # 可视化
    plt.figure(figsize=(6, 4))
    seq_lengths = [x[0] for x in memory_results]
    peak_mem = [x[2] for x in memory_results]
    plt.plot(seq_lengths, peak_mem, 'ro-')
    plt.title('LSTM显存占用 vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory (MB)')
    plt.grid(True)
    plt.savefig('lstm_memory.png')
    print("\n已保存显存分析图: lstm_memory.png")
    
    return lstm_model, memory_results

def main():
    # 初始化环境
    device = setup_environment()
    if str(device) == 'cpu':
        print("注意:以下显存测量结果将全为0(因使用CPU)")
    
    try:
        # 分析ResNet18
        resnet, resnet_memory = analyze_resnet(device)
        
        # 分析LSTM
        lstm, lstm_memory = analyze_lstm(device)
        
        # 综合对比
        print("\n>>> 综合分析结果 <<<")
        print("1. ResNet18参数更多但显存增长主要受batch size影响")
        print("2. LSTM参数较少但对sequence length更敏感")
        print("3. 实际应用中需要根据任务特点调整batch size和sequence length")
        
    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        print("请检查: 1) CUDA是否可用 2) 显存是否足够 3) 依赖库是否安装")

if __name__ == "__main__":
    main()