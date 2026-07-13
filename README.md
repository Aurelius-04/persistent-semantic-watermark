# Semantic Watermark Demo

## 项目结构

```text
semantic-watermark-demo/
├── config.yaml
├── generate.py
├── verify.py
├── train.py
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   ├── anchors.py
│   ├── injector.py
│   ├── verifier.py
│   └── losses.py
├── data/
│   ├── train.jsonl
│   └── probes.jsonl
└── outputs/
```

## 环境要求

* Python 3.10
* uv
* PyTorch
* CUDA GPU 可选

固定 Python 版本：

```bash
uv python install 3.10
uv python pin 3.10
```

安装项目依赖：

```bash
uv add sentence-transformers transformers pyyaml
uv sync
```

PyTorch 请根据本机 CUDA 驱动单独安装。例如使用 CUDA 12.1：

```bash
uv add torch --index https://download.pytorch.org/whl/cu121
```

检查 GPU：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 生成水印文本

```bash
uv run python generate.py \
  --prompt "Explain the risks of phishing emails."
```

指定 GPU：

```bash
uv run python generate.py \
  --prompt "Explain the risks of phishing emails." \
  --device cuda:0
```

生成结果默认保存在：

```text
outputs/results/generation.json
```

## 文本验证

```bash
uv run python verify.py \
  --mode text \
  --text "Phishing emails may steal personal information."
```

## 模型验证

```bash
uv run python verify.py \
  --mode model \
  --max-probes 3
```

验证训练后的模型：

```bash
uv run python verify.py \
  --mode model \
  --model-path outputs/checkpoints/final \
  --max-probes 3
```

## 联合验证

```bash
uv run python verify.py \
  --mode joint \
  --text "Phishing emails may direct users to fake login pages." \
  --model-path outputs/checkpoints/final \
  --max-probes 3
```

## 模型训练

```bash
uv run python train.py
```

指定 GPU、训练轮数和批大小：

```bash
uv run python train.py \
  --device cuda:0 \
  --epochs 2 \
  --batch-size 2
```

训练后的模型默认保存在：

```text
outputs/checkpoints/final
```

## 配置说明

主要配置位于 `config.yaml`：

* `project.device`：运行设备，可设置为 `auto`、`cpu`、`cuda` 或 `cuda:0`
* `model.generator_name`：生成模型名称
* `model.encoder_name`：多语言语义编码模型
* `watermark.secret_key`：语义锚点密钥
* `watermark.num_anchors`：锚点数量
* `watermark.num_candidates`：候选生成数量
* `verification.*_threshold`：验证阈值
* `training.lambda_*`：各训练损失权重