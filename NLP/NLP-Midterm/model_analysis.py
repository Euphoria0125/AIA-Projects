import torch
import torch.nn as nn
import torchvision.models as models
from pympler import asizeof
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']

def setup_environment():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n{'='*50}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("\n警告：CUDA不可用，将使用CPU运行")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"{'='*50}\n")
    torch.cuda.empty_cache()
    return device

def analyze_resnet(device):
    print("\n>>> 开始ResNet18分析 <<<")
    resnet18 = models.resnet18(pretrained=False).to(device)

    def count_parameters(model):
        table, total = [], 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                table.append((name, param_count))
                total += param_count
        return table, total

    print("\n[ResNet18结构概览]")
    print(resnet18)

    params_table, total_params = count_parameters(resnet18)
    print("\n[各层参数量明细]")
    for name, params in params_table[:5]:
        print(f"{name:50} {params:>12,}")
    print(f"...(中间层省略)...")
    for name, params in params_table[-3:]:
        print(f"{name:50} {params:>12,}")
    print(f"\n总可训练参数量: {total_params:,}")

    conv1 = resnet18.conv1
    in_ch, out_ch = conv1.in_channels, conv1.out_channels
    k_size = conv1.kernel_size[0]
    theory = (k_size * k_size * in_ch) * out_ch + (out_ch if conv1.bias is not None else 0)
    actual = sum(p.numel() for p in conv1.parameters())
    print(f"\n[参数量验证] 第一卷积层: 理论值={theory:,}, 实际值={actual:,}, 验证{'通过' if theory==actual else '失败'}")

    def test_memory(batch_sizes=[1, 4, 8, 16]):
        print("\n[显存占用测试]")
        results = []
        for bs in batch_sizes:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(bs, 3, 224, 224).to(device)
            torch.cuda.synchronize()
            init_mem = torch.cuda.memory_allocated()
            _ = resnet18(dummy_input)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            results.append((bs, init_mem / 1024**2, peak_mem / 1024**2))
            print(f"Batch Size {bs:2}: 初始={init_mem/1024**2:6.2f}MB, 峰值={peak_mem/1024**2:6.2f}MB")
        return results

    memory_results = test_memory()
    plt.figure(figsize=(6, 4))
    plt.plot([x[0] for x in memory_results], [x[2] for x in memory_results], 'bo-')
    plt.title('ResNet18显存占用 vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory (MB)')
    plt.grid(True)
    plt.savefig('data/resnet_memory.png')
    print("\n已保存显存分析图: resnet_memory.png")
    return resnet18, memory_results

def analyze_lstm(device):
    print("\n>>> 开始LSTM分析 <<<")
    class LSTMModel(nn.Module):
        def __init__(self, input_size=100, hidden_size=256, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
            self.fc = nn.Linear(hidden_size, 10)
        def forward(self, x):
            x, _ = self.lstm(x)
            return self.fc(x[-1, :, :])

    lstm_model = LSTMModel().to(device)
    theory = 4*(100*256 + 256*256 + 256) + 4*(256*256 + 256*256 + 256) + (256*10 + 10)
    actual = sum(p.numel() for p in lstm_model.parameters())
    print(f"\n理论参数量: {theory:,}\n实际参数量: {actual:,}\n差异: {actual - theory}")

    memory_results = []
    for seq_len in [50, 100, 200, 300]:
        torch.cuda.empty_cache()
        dummy_input = torch.randn(seq_len, 32, 100).to(device)
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        with torch.no_grad():
            _ = lstm_model(dummy_input)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        memory_results.append((seq_len, start_mem / 1024**2, peak_mem / 1024**2))
    plt.figure(figsize=(6, 4))
    plt.plot([x[0] for x in memory_results], [x[2] for x in memory_results], 'ro-')
    plt.title('LSTM显存占用 vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory (MB)')
    plt.grid(True)
    plt.savefig('data/lstm_memory.png')
    print("\n已保存显存分析图: lstm_memory.png")
    return lstm_model, memory_results

def analyze_bert(device):
    print("\n>>> 开始BERT分析 <<<")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("\n[BERT结构概览]")
    print(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total:,}\n可训练参数量: {trainable:,}")

    vocab_size = model.embeddings.word_embeddings.num_embeddings
    dim = model.embeddings.word_embeddings.embedding_dim
    print(f"\n[Embedding层参数验证] 理论: {vocab_size * dim:,}, 实际: {model.embeddings.word_embeddings.weight.numel():,}")

    memory_results = []
    for seq_len in [8, 32, 64, 128, 256]:
        torch.cuda.empty_cache()
        dummy_input = tokenizer("This is a test input.", return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len)
        dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        with torch.no_grad():
            _ = model(**dummy_input)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        memory_results.append((seq_len, start_mem / 1024**2, peak_mem / 1024**2))
        print(f"Seq Len {seq_len}: 初始={start_mem/1024**2:.2f}MB, 峰值={peak_mem/1024**2:.2f}MB")

    plt.figure(figsize=(6, 4))
    plt.plot([x[0] for x in memory_results], [x[2] for x in memory_results], 'go-')
    plt.title('BERT显存占用 vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory (MB)')
    plt.grid(True)
    plt.savefig('data/bert_memory.png')
    print("\n已保存显存分析图: bert_memory.png")
    return model, memory_results

def main():
    device = setup_environment()
    try:
        resnet, _ = analyze_resnet(device)
        lstm, _ = analyze_lstm(device)
        bert, _ = analyze_bert(device)
        print("\n>>> 分析完成 <<<")
    except Exception as e:
        print(f"\n错误发生: {e}\n请检查 CUDA 是否可用、显存是否足够、依赖库是否安装")

if __name__ == "__main__":
    main()