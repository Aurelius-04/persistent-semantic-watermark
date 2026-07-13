# Persistent Semantic Watermark

## Project Structure

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

## Requirements

* Python 3.10
* uv
* PyTorch
* CUDA-capable GPU optional

Install and pin Python 3.10:

```bash
uv python install 3.10
uv python pin 3.10
```

Install the project dependencies:

```bash
uv add sentence-transformers transformers pyyaml
uv sync
```

Install PyTorch separately according to your local CUDA driver version. For example, to use CUDA 12.1:

```bash
uv add torch --index https://download.pytorch.org/whl/cu121
```

Check GPU availability:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Generate Watermarked Text

```bash
uv run python generate.py \
  --prompt "Explain the risks of phishing emails."
```

Specify a GPU device:

```bash
uv run python generate.py \
  --prompt "Explain the risks of phishing emails." \
  --device cuda:0
```

By default, the generated result is saved to:

```text
outputs/results/generation.json
```

## Text Verification

```bash
uv run python verify.py \
  --mode text \
  --text "Phishing emails may steal personal information."
```

## Model Verification

```bash
uv run python verify.py \
  --mode model \
  --max-probes 3
```

Verify a trained model:

```bash
uv run python verify.py \
  --mode model \
  --model-path outputs/checkpoints/final \
  --max-probes 3
```

## Joint Verification

```bash
uv run python verify.py \
  --mode joint \
  --text "Phishing emails may direct users to fake login pages." \
  --model-path outputs/checkpoints/final \
  --max-probes 3
```

## Model Training

```bash
uv run python train.py
```

Specify the GPU device, number of training epochs, and batch size:

```bash
uv run python train.py \
  --device cuda:0 \
  --epochs 2 \
  --batch-size 2
```

By default, the trained model is saved to:

```text
outputs/checkpoints/final
```

## Configuration

The main configuration options are defined in `config.yaml`:

* `project.device`: Runtime device. Supported values include `auto`, `cpu`, `cuda`, and `cuda:0`.
* `model.generator_name`: Name of the text generation model.
* `model.encoder_name`: Name of the multilingual semantic encoder model.
* `watermark.secret_key`: Secret key used to generate semantic anchors.
* `watermark.num_anchors`: Number of semantic anchors.
* `watermark.num_candidates`: Number of candidate outputs to generate.
* `verification.*_threshold`: Verification thresholds.
* `training.lambda_*`: Weight assigned to each training loss term.
