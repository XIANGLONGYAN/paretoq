# ParetoQ - 低比特量化感知训练 (QAT) 框架

ParetoQ 是一个基于 HuggingFace Transformers 的低比特量化感知训练框架，支持 **1/2/3/4 bit** 权重量化，适用于 LLaMA、Qwen2、Mistral 等主流大语言模型。

核心特性：
- 支持 1-bit（Binary）、2-bit（Ternary）、3-bit、4-bit 权重量化
- 基于 Learned Step-size Quantization (LSQ) 的可学习量化步长
- 支持 Stretched Elastic Quantization（2-bit 场景优化）
- 兼容 HuggingFace `Trainer` API，支持 DeepSpeed、FSDP 等分布式训练
- 内置 PPL（Perplexity）评估和 lm_eval zero-shot 评估

---

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [BF16 Baseline 评估](#bf16-baseline-评估)
- [QAT 训练](#qat-训练)
- [加载量化模型评估](#加载量化模型评估)
- [项目结构](#项目结构)
- [关键参数说明](#关键参数说明)
- [常见问题 (FAQ)](#常见问题-faq)

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n paretoq python=3.11 -y
conda activate paretoq
```

### 2. 安装 PyTorch

> 💡 **建议使用 CUDA 12.4**，但请根据自己的 GPU 型号和驱动版本适配安装对应的 PyTorch 版本。可参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择合适的版本。

```bash
# 示例：CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装依赖

所有依赖已统一写在 `requirement.txt` 中（包括 lm_eval、wandb 等），一条命令安装：

```bash
pip install -r requirement.txt
```

### 4. 登录 wandb（用于训练日志可视化）

```bash
wandb login  # 输入你的 API key
```

### 5. 验证安装

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA (compiled): {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

预期输出：
```
PyTorch: 2.6.0+cu124
CUDA (compiled): 12.4
CUDA available: True
GPU: NVIDIA H20
```

---

## 数据准备

使用 `gen_data.py` 从 HuggingFace 下载数据集并转为 jsonl 格式。

### 生成 wikitext-2 训练集和测试集

```bash
# 将下面的路径修改为你自己的数据存放路径
DATA_DIR=/path/to/your/dataset/paretoq_data

# 生成训练集
python gen_data.py \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir $DATA_DIR \
    --splits train

# 生成测试集
python gen_data.py \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir $DATA_DIR \
    --splits test
```

### 生成其他数据集（可选）

```bash
# wikitext-103（更大的训练集，约 100M tokens）
python gen_data.py \
    --dataset_name wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --output_dir $DATA_DIR \
    --splits train test

# C4 数据集（取前 50000 条）
python gen_data.py \
    --dataset_name allenai/c4 \
    --dataset_config en \
    --output_dir $DATA_DIR \
    --splits train \
    --max_samples 50000
```

### 生成的数据文件

```
$DATA_DIR/
├── wikitext_wikitext-2-raw-v1_train.jsonl   # wikitext-2 训练数据
├── wikitext_wikitext-2-raw-v1_test.jsonl    # wikitext-2 测试数据
└── ...
```

每行为一个 JSON 对象，格式为 `{"text": "...文本内容..."}`。

### gen_data.py 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_name` | HuggingFace 数据集名 | `wikitext` |
| `--dataset_config` | 数据集配置/子集 | `wikitext-2-raw-v1` |
| `--output_dir` | 输出目录 | `/tmp/paretoq_data` |
| `--splits` | 下载的数据分割（支持多个） | `train test` |
| `--min_length` | 最短文本字符数（过滤短文本） | `10` |
| `--max_samples` | 每个 split 最大样本数 | `None`（全部） |

---

## BF16 Baseline 评估

在做量化训练之前，先评估模型在 BF16（全精度）下的性能，作为 baseline 参考。

### 运行 BF16 评估

```bash
cd /path/to/paretoq
source scripts/eval_bf16.sh
```

### eval_bf16.sh 脚本内容

```bash
# 请将以下路径修改为你自己的实际路径
MODEL_DIR=/path/to/pretrained/models
DATA_DIR=/path/to/your/dataset/paretoq_data

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29505 train.py \
--local_dir "/tmp/llama/" \
--input_model_filename "${MODEL_DIR}/llama-3.2-1B" \
--output_model_filename "1B-bf16-baseline" \
--train_data_local_path "${DATA_DIR}/wikitext_wikitext-2-raw-v1_train.jsonl" \
--eval_data_local_path "${DATA_DIR}/wikitext_wikitext-2-raw-v1_test.jsonl" \
--do_train False \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "none" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits 16 \
--eval_ppl \
--eval_lm_eval \
--tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge" \
--eval_batch_size 16
```

**关键参数说明：**
- `--do_train False`：不训练，只评估
- `--w_bits 16`：不量化（BF16 全精度）
- `--eval_ppl`：评估 wikitext2 和 c4 的 Perplexity
- `--eval_lm_eval`：运行 lm_eval zero-shot 评估
- `--tasks`：指定 lm_eval 评估任务

### 评估输出示例

```
wikitext2 perplexity: 9.xx
c4 perplexity: 11.xx
============================================================
Summary:
  piqa: 75.xx%
  hellaswag: 55.xx%
  winogrande: 60.xx%
  arc_easy: 65.xx%
  arc_challenge: 35.xx%
  Average Acc: 58.xx%
============================================================
```

---

## QAT 训练

### 运行 QAT 训练

```bash
cd /path/to/paretoq
source scripts/train.sh
```

### train.sh 脚本内容

```bash
# 请将以下路径修改为你自己的实际路径
MODEL_DIR=/path/to/pretrained/models
DATA_DIR=/path/to/your/dataset/paretoq_data

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29511 train.py \
--local_dir "/tmp/llama/" \
--input_model_filename "${MODEL_DIR}/llama-3.2-1B" \
--output_model_filename "1B-finetuned-4bit" \
--train_data_local_path "${DATA_DIR}/wikitext_wikitext-2-raw-v1_train.jsonl" \
--eval_data_local_path "${DATA_DIR}/wikitext_wikitext-2-raw-v1_test.jsonl" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/current \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "wandb" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--dataloader_num_workers 0 \
--seed 42 \
--w_bits 4 \
--eval_ppl \
--eval_lm_eval \
--tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge" \
--eval_batch_size 64
```

### 关键训练参数

| 参数 | 说明 |
|------|------|
| `--w_bits` | 量化位数（1/2/3/4），16 为不量化 |
| `--qat True` | 启用量化感知训练模式 |
| `--do_train True` | 开启训练 |
| `--num_train_epochs 1` | 训练轮数 |
| `--per_device_train_batch_size 8` | 每 GPU 训练 batch size |
| `--learning_rate 2e-5` | 学习率 |
| `--lr_scheduler_type "cosine"` | 学习率调度策略 |
| `--model_max_length 2048` | 最大序列长度 |
| `--dataloader_num_workers 0` | DataLoader 线程数（设为 0 保证可复现性） |
| `--seed 42` | 随机种子（保证可复现性） |
| `--eval_ppl` | 训练后评估 PPL |
| `--eval_lm_eval` | 训练后运行 lm_eval |

### 训练流程说明

1. **加载预训练模型**：以 BF16 精度加载原始 HuggingFace 模型
2. **替换 Linear 层**：将 `nn.Linear` 替换为 `QuantizeLinear`（跳过 `lm_head` 和 `embed` 层）
3. **初始化量化步长**：根据权重分布自动计算 `weight_clip_val`（量化步长 α）
4. **QAT 训练**：使用 STE（Straight-Through Estimator）反向传播，同时学习权重和量化步长
5. **评估**：训练结束后自动运行 PPL 和 zero-shot 评估

### 训练输出

模型保存路径：`/tmp/llama/models/<output_model_filename>/`

保存的文件包括：
- `pytorch_model.bin` 或 `model.safetensors`：包含量化后的权重和学习到的 `weight_clip_val`
- `config.json`：模型配置
- `tokenizer` 相关文件

---

## 项目结构

```
paretoq/
├── train.py                    # 主训练/评估入口
├── gen_data.py                 # 数据生成脚本
├── requirement.txt             # Python 依赖
├── README.md                   # 本文档
├── scripts/
│   ├── train.sh                # QAT 训练脚本
│   ├── eval_bf16.sh            # BF16 baseline 评估脚本
│   └── eval.sh                 # 量化模型评估脚本
└── utils/
    ├── __init__.py
    ├── utils.py                # 通用工具（set_seed, logger, save_model）
    ├── utils_quant.py          # 量化核心模块（QuantizeLinear, replace_linear_with_quantized）
    ├── datautils.py            # 数据加载（CustomJsonDataset, get_loaders, PPL数据）
    ├── process_args.py         # 参数定义与解析
    └── eval.py                 # 评估逻辑（PPL + lm_eval）
```

### 核心模块说明

| 模块 | 功能 |
|------|------|
| `utils/utils_quant.py` | 量化 Linear 层实现，包含 `LsqBinaryTernaryExtension`（1/3/4-bit）和 `StretchedElasticQuant`（2-bit）两种量化方法 |
| `utils/datautils.py` | 训练数据处理（jsonl → tokenized → grouped chunks）和 PPL 评估数据加载 |
| `utils/eval.py` | PPL 评估（wikitext2, c4）和 lm_eval zero-shot 评估（piqa, hellaswag 等） |
| `utils/process_args.py` | 命令行参数定义：`ModelArguments`, `DataArguments`, `TrainingArguments`, `EvalArguments` |

---

## 关键参数说明

### 量化位数 (`--w_bits`)

| 值 | 含义 | 量化方法 |
|----|------|----------|
| 1 | 1-bit Binary（-α, +α） | `LsqBinaryTernaryExtension`，sign 函数 |
| 2 | 2-bit Ternary | `StretchedElasticQuant`，拉伸弹性量化 |
| 3 | 3-bit（-4 ~ +3）| `LsqBinaryTernaryExtension`，round + clamp |
| 4 | 4-bit（-8 ~ +7）| `LsqBinaryTernaryExtension`，round + clamp |
| 16 | 不量化（BF16 全精度）| 不替换 Linear 层 |

### 可复现性设置

为确保两次训练结果完全一致，在训练脚本中添加以下参数即可：

```bash
--seed 42 \
--full_determinism True \
--dataloader_num_workers 0 \
```

**说明：**

| 参数 | 作用 |
|------|------|
| `--seed 42` | 固定所有随机种子（Python、NumPy、PyTorch） |
| `--full_determinism True` | 启用完全确定性模式，内部会设置 `FLASH_ATTENTION_DETERMINISTIC=1`、`CUDA_LAUNCH_BLOCKING=1`、`torch.use_deterministic_algorithms(True)` 等 |
| `--dataloader_num_workers 0` | 避免多线程数据加载引入的顺序不确定性 |

> 💡 设置了 `--full_determinism True` 后，无需在代码中手动调用 `set_seed()` 或设置 `cudnn.deterministic` 等，Trainer 会自动处理所有确定性相关设置。

---

## 常见问题 (FAQ)

### Q1: 评估时出现 SIGFPE (Signal 8) 崩溃

**原因**：PyTorch cu121 的 SDPA kernel 在 Hopper 架构 GPU（H20/H100）上有浮点异常 bug。

**解决方案**：升级到 PyTorch cu124：
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Q2: 训练时显存不足 (OOM)

- 减小 `--per_device_train_batch_size`（如 4 或 2）
- 开启梯度检查点：`--gradient_checkpointing True`
- 减小 `--model_max_length`（如 1024）

---

## License

BSD-style License. See [LICENSE](./LICENSE) for details.
