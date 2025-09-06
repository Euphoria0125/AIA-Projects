# NLP Final Project, AIA, HUST, 2025

本项目包含两个部分：  

1. 图像描述生成  
2. 图像风格迁移  

---

## 目录结构

```
.
├─ Qwen-VL/
│  ├─ finetune/
│  │  ├─ data/                  # 数据集文件夹（Train/Val 图像 + train.json/eval.json）
│  │  └─ output/                # 训练或推理输出（如模型权重）
│  ├─ result/
│  │  └─ pretrained/            # 推理与评测结果
│  │     ├─ output.json         # inference.py 生成的图像描述结果
│  │     └─ metrics.json        # metrics.py 生成的评价指标
│  ├─ interface.py              # 定义图像描述生成的 prompt
│  ├─ inference.py              # 推理脚本，调用 Qwen-VL 生成描述
│  ├─ metrics.py                # 评测脚本，计算 BLEU/CIDEr/ROUGE/SPICE/BERTScore
│  ├─ environment.yaml          # conda 环境依赖
│  └─ ...
├─ stable-diffusion/
│  ├─ models/
│  │  └─ ldm/stable-diffusion-v1/
│  │     └─ model.ckpt          # Stable Diffusion v1-4 模型权重
│  ├─ scripts/
│  │  └─ img2img.py             # 风格迁移核心脚本
│  ├─ prompt.txt                # 存放风格迁移的提示词，每行一个
│  ├─ outputs/
│  │  └─ img2img/               # 风格迁移生成结果保存目录
│  ├─ environment.yaml          # conda 环境依赖
│  └─ ...
└─ README.md
```

---

## 一、图像描述生成

### 模型与设置

- 使用 **Qwen-VL 多模态大模型**，对未标注图片进行描述生成。  
- 通过 Prompt Engineering 优化输入提示，提升输出质量。  
- 仅使用预训练模型进行推理，不进行训练。  

### 模型与数据来源

- **模型权重**：请在 [百度网盘链接（提取码：jg59）](https://pan.baidu.com/share/init?surl=_cCNee6uniTtP3c3ZUcgJA&pwd=jg59) 下载，位于 `NLP Final Project/MiniCPM-V/LoRA权重/output`。  
  下载完成后请将 `output` 文件夹置于 `finetune/` 目录下。由于文件较大，可以只下载以下必要文件：
  - `checkpoint-5000/`
  - `adapter_model.safetensors`
  - `tokenizer_config.json`
  - `trainer_state.json`
  - `training_args.bin`
  - `adapter_config.json`
  - `special_tokens_map.json`
  - `tokenizer.model`

- **数据集**：课程提供的图片数据集，组织为 `finetune/data/` 下的 `Train/`, `Val/` 文件夹和对应的 `train.json`, `eval.json` 标注文件。  

### 环境配置

```bash
git clone <your_repo_url>
cd <your_repo_root>
conda env create -f Qwen-VL/environment.yaml
conda activate MiniCPMV
```

### Prompt 设置

在 `interface.py` 中定义 prompt，例如：

```python
question = '''
请根据输入图像，生成一段条理清晰、结构完整的客观描述。要求：
1. 说明主要对象及位置，点明空间关系。
2. 详细描绘对象的颜色、材质、装饰和纹理。
3. 补充环境与次要元素，体现场景层次。
4. 交代光线与整体色调，并总结氛围。
5. 使用自然连贯的段落表述。
6. 保持客观、中性。
'''
```

### 数据与运行

数据集需放置于 `finetune/data/` 下，格式包括：

- `Train/`, `Val/`：图片目录  
- `train.json`, `eval.json`：图文对标注  

运行推理与评测：

```bash
cd Qwen-VL
python inference.py    # 生成描述
python metrics.py      # 计算指标
```

结果文件：

- `result/pretrained/output.json`：模型生成的图像描述  
- `result/pretrained/metrics.json`：自动评估结果（BLEU、CIDEr、ROUGE、SPICE、BERTScore）  

---

## 二、图像风格迁移

### 模型与说明

- 使用 **Stable Diffusion v1-4** 进行风格迁移。  
- Stable Diffusion v1-4 是一种潜在扩散模型 (Latent Diffusion Model, LDM)，包含下采样因子为 8 的自编码器、约 8.6 亿参数的 UNet，以及 CLIP ViT-L/14 文本编码器。  
- 模型在 **LAION-2B-en** 和 **LAION-Aesthetics v2 5+** 数据集上预训练与微调得到。  
- 在保留原图结构的前提下，通过文本提示实现风格化。

### 模型与数据来源

- **模型权重**：来自 CompVis 在 Hugging Face 提供的 [Stable Diffusion v1-4 权重](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)，需遵循 CreativeML Open RAIL-M 许可协议。  
- **数据集**：风格迁移部分不依赖额外标注，仅需输入待迁移的图片。  
- 将下载的 `sd-v1-4.ckpt` 重命名为 `model.ckpt`，放置在：  
  `stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt`  

### 环境配置

```bash
cd <your_repo_root>
conda env create -f stable-diffusion/environment.yaml
conda activate ldm
```

### 提示词文件

`prompt.txt` 示例：

```
traditional Chinese ink painting, brush strokes on rice paper, minimalistic landscape
impressionist oil painting, Claude Monet inspired, glowing light reflections
cyberpunk cityscape, neon lights, futuristic atmosphere
```

### 运行

```bash
cd stable-diffusion
python scripts/img2img.py ^
  --init-img "<path_to_image>" ^
  --from-file prompt.txt ^
  --strength 0.55 ^
  --ddim_steps 100 ^
  --n_iter 1 ^
  --n_samples 1 ^
  --scale 7.5 ^
  --outdir outputs/img2img
```

生成结果保存在 `outputs/img2img/samples/`。

### 参数说明

- `strength`：原图保真度与风格化程度的平衡，常用 0.3–0.6  
- `ddim_steps`：步数，数值越高越精细，推荐 50–100  
- `scale`：文本引导强度，推荐 7–10  
- 输入图像需调整为 32 的倍数，最长边不超过 1024  

---
