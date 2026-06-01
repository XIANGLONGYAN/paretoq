# Muon 优化器使用与集成说明 


## 1. 快速使用指南 (Quick Start)

如果要在微调/量化感知训练（QAT）中启用 Muon 优化器，无需修改 Python 代码，只需在启动脚本 `train.sh` 中配置以下三个参数即可：

```bash
# 1. 开启 Muon 优化器开关
--use_muon True \

# 2. 设置 AdamW 轨道的安全初始学习率（用于 Embedding, LM Head, LayerNorms 等 AdamW 的学习率）
--learning_rate 2e-5 \

# 3. 设置 Muon 轨道的初始学习率
--muon_learning_rate 2e-4 \
```


## 2. 核心设计与分流原理 (Technical Design)

为了保障大模型在微调过程中的稳定性，采用了 **“双轨道参数分流与自适应学习率”** 机制。`MuonTrainer` 会在底层将模型参数拆分为两组独立优化：

```
                           ┌──► 隐藏层 2D 权重矩阵 ──► Muon 优化器 ──► 高学习率
                           │    (Attention, MLP 投影层)
模型全部参数 (Parameters) ──┤
                           │
                           └──► 其余一维/低维参数 ──► AdamW 优化器 ──► 低学习率
                                (Embedding, Head, LayerNorm, 偏置, 量化截断值)
```

1.  **Muon 轨道 (use_muon=True)**：
    *   **适用对象**：维度 $\ge 2$ 的隐藏层权重矩阵（排除 Embedding 与 LM Head）。
    *   **原理**：通过 Newton-Schulz 迭代进行矩阵正交化更新。由于更新尺度被拉平为谱范数 `1`，它需要较大的初始学习率来驱动正交基旋转。
2.  **AdamW 轨道 (use_muon=False)**：
    *   **适用对象**：词表层（Embedding & Head）、一维偏置、所有归一化层参数（LayerNorm/RMSNorm）以及量化截断值。
    *   **原理**：使用传统的 AdamW 进行自适应坐标轴缩放更新。这些敏感参数如果直接使用较大的步长，模型会在第一步发生梯度爆炸直接崩溃（NaN），因此必须使用温和的学习率。

---

## 3. 分布式与多卡适配 (Distributed Compatibility)

训练器会根据运行环境，自动选择性能最佳的 Muon 实现，无需手动调节：
*   **多卡 DDP 环境**：自动使用 `MuonWithAuxAdam`，支持跨 GPU 的通信规约（`all_gather`）以保持权重同步。
*   **单卡/单进程环境**：自动降级使用 `SingleDeviceMuonWithAuxAdam`，避免任何不必要的通信开销。

---

## 4. 代码改动清单 (Code Changelog / File Changes)

### 📂 [utils/process_args.py]
在 `TrainingArguments` 参数类中新增了两个控制参数：
```python
    qat: Optional[bool] = field(default=False)
    # === 新增以下两个参数 ===
    use_muon: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use Muon optimizer. When True, bypasses native optim validation."}
    )
    muon_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for Muon optimizer. If not set, defaults to learning_rate * 10."}
    )
```

### 📂 [train.py]
1. **定义了新子类 `MuonTrainer`** 并覆盖 `create_optimizer` 方法进行参数分流：
```python
class MuonTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if not self.args.use_muon:
            return super().create_optimizer()

        from optimizer.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

        # 隐藏层 2D 矩阵用 Muon
        hidden_matrix_params = [
            p for n, p in self.model.named_parameters() 
            if p.ndim >= 2 and "embed" not in n and "lm_head" not in n
        ]
        # 低维/一维参数、Embedding、Final Head 用 AdamW
        other_params = [
            p for n, p in self.model.named_parameters() 
            if p.ndim < 2 or "embed" in n or "lm_head" in n
        ]

        muon_lr = (
            self.args.muon_learning_rate 
            if self.args.muon_learning_rate is not None 
            else self.args.learning_rate * 10
        )

        param_groups = [
            {
                "params": hidden_matrix_params,
                "lr": muon_lr,
                "momentum": 0.95,
                "weight_decay": self.args.weight_decay,
                "use_muon": True,
            },
            {
                "params": other_params,
                "lr": self.args.learning_rate,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": self.args.weight_decay,
                "use_muon": False,
            }
        ]

        if dist.is_initialized():
            self.optimizer = MuonWithAuxAdam(param_groups)
        else:
            self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        return self.optimizer
```
2. **将训练器替换为我们的子类**（约在文件第 140 行）：
```python
-    myTrainer = Trainer
+    myTrainer = MuonTrainer
```

---

## 5. 致谢 (Acknowledgements)

本模块集成的 Muon 优化器源自 Keller Jordan 的开源项目 [Muon](https://kellerjordan.github.io/posts/muon/)。特别鸣谢开源社区贡献者 `@scottjmaddox`（批处理并行实现）以及 `@YouJiacheng`、`@jxbz`、`@leloykun` 等人进行的算法与工程优化。
