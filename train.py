import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder
from models.losses import WatermarkLoss


class WatermarkDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        prompt_field: str,
        response_field: str,
        max_length: int,
    ) -> None:
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        data_path = Path(path)

        if not data_path.exists():
            raise FileNotFoundError(f"训练数据不存在：{data_path}")

        with data_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{data_path} 第 {line_number} 行不是有效 JSON"
                    ) from error

                prompt = item.get(prompt_field)
                response = item.get(response_field)

                if not isinstance(prompt, str) or not prompt.strip():
                    continue

                if not isinstance(response, str) or not response.strip():
                    continue

                sample = self._encode_sample(
                    prompt.strip(),
                    response.strip(),
                )

                if sample is not None:
                    self.samples.append(sample)

        if not self.samples:
            raise ValueError("训练数据中没有可用样本")

    def _encode_sample(
        self,
        prompt: str,
        response: str,
    ) -> dict[str, torch.Tensor] | None:
        prefix = f"{prompt}\n\n"
        full_text = prefix + response

        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        prefix_encoding = self.tokenizer(
            prefix,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        prefix_length = min(
            len(prefix_encoding["input_ids"]),
            len(input_ids),
        )

        labels = list(input_ids)

        for index in range(prefix_length):
            labels[index] = -100

        if all(label == -100 for label in labels):
            return None

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        return self.samples[index]


class WatermarkCollator:
    def __init__(
        self,
        pad_token_id: int,
    ) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        samples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(
            sample["input_ids"].shape[0]
            for sample in samples
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for sample in samples:
            current_length = sample["input_ids"].shape[0]
            padding_length = max_length - current_length

            input_ids = torch.cat(
                [
                    sample["input_ids"],
                    torch.full(
                        (padding_length,),
                        self.pad_token_id,
                        dtype=torch.long,
                    ),
                ]
            )

            attention_mask = torch.cat(
                [
                    sample["attention_mask"],
                    torch.zeros(
                        padding_length,
                        dtype=torch.long,
                    ),
                ]
            )

            labels = torch.cat(
                [
                    sample["labels"],
                    torch.full(
                        (padding_length,),
                        -100,
                        dtype=torch.long,
                    ),
                ]
            )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


class SemanticProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(hidden_states)


def load_config(path: str) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("配置文件内容无效")

    return config


def resolve_device(device_name: str) -> torch.device:
    device_name = str(device_name).strip().lower()

    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用 CUDA GPU")

        return torch.device("cuda")

    if device_name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用 CUDA GPU")

        try:
            index = int(device_name.split(":", maxsplit=1)[1])
        except ValueError as error:
            raise ValueError(
                f"无效的设备配置：{device_name}"
            ) from error

        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"请求使用 cuda:{index}，"
                f"当前只有 {torch.cuda.device_count()} 张 GPU"
            )

        return torch.device(device_name)

    raise ValueError(f"不支持的设备配置：{device_name}")


def resolve_dtype(
    precision: str,
    device: torch.device,
) -> torch.dtype:
    precision = str(precision).strip().lower()

    if device.type != "cuda":
        return torch.float32

    if precision == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    if precision == "fp16":
        return torch.float16

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 BF16")

        return torch.bfloat16

    if precision == "fp32":
        return torch.float32

    raise ValueError(f"不支持的精度配置：{precision}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_runtime(
    config: dict,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        return

    runtime_config = config.get("runtime", {})

    allow_tf32 = bool(
        runtime_config.get("allow_tf32", True)
    )

    cudnn_benchmark = bool(
        runtime_config.get("cudnn_benchmark", False)
    )

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark


def freeze_generator(
    model: nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    output_embeddings = model.get_output_embeddings()

    if output_embeddings is not None:
        for parameter in output_embeddings.parameters():
            parameter.requires_grad = True

    if hasattr(model, "transformer"):
        transformer = model.transformer

        if hasattr(transformer, "h") and len(transformer.h) > 0:
            for parameter in transformer.h[-1].parameters():
                parameter.requires_grad = True

        if hasattr(transformer, "ln_f"):
            for parameter in transformer.ln_f.parameters():
                parameter.requires_grad = True

    elif hasattr(model, "model"):
        base_model = model.model

        if hasattr(base_model, "layers") and len(base_model.layers) > 0:
            for parameter in base_model.layers[-1].parameters():
                parameter.requires_grad = True

        if hasattr(base_model, "norm"):
            for parameter in base_model.norm.parameters():
                parameter.requires_grad = True


def build_semantic_trajectories(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    projection: SemanticProjection,
    span_size: int,
    span_stride: int,
) -> list[torch.Tensor]:
    trajectories = []

    for sample_index in range(hidden_states.shape[0]):
        response_mask = labels[sample_index] != -100
        response_states = hidden_states[sample_index][response_mask]

        if response_states.shape[0] == 0:
            trajectories.append(
                torch.empty(
                    0,
                    projection.network[-1].out_features,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            )
            continue

        span_vectors = []
        start = 0

        while start < response_states.shape[0]:
            end = min(
                start + span_size,
                response_states.shape[0],
            )

            span_state = response_states[start:end].mean(dim=0)
            span_vectors.append(span_state)

            if end >= response_states.shape[0]:
                break

            start += span_stride

        stacked_states = torch.stack(span_vectors)
        projected = projection(stacked_states)
        trajectories.append(projected)

    return trajectories


def get_autocast_context(
    device: torch.device,
    dtype: torch.dtype,
):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(
            device,
            non_blocking=device.type == "cuda",
        )
        for key, value in batch.items()
    }


def save_checkpoint(
    model,
    tokenizer,
    projection: SemanticProjection,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: Path,
    config: dict,
) -> None:
    checkpoint_dir = output_dir / f"epoch-{epoch + 1}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    projection_data = {
        "state_dict": projection.state_dict(),
        "input_dim": projection.network[0].in_features,
        "output_dim": projection.network[-1].out_features,
        "epoch": epoch + 1,
        "global_step": global_step,
    }

    torch.save(
        projection_data,
        checkpoint_dir / "semantic_projection.pt",
    )

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
        },
        checkpoint_dir / "training_state.pt",
    )

    with (
        checkpoint_dir / "training_config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练持久语义水印演示模型"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.device is not None:
        config["project"]["device"] = args.device

    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs 必须大于 0")

        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size 必须大于 0")

        config["training"]["batch_size"] = args.batch_size

    device = resolve_device(
        config["project"]["device"]
    )

    dtype = resolve_dtype(
        config["runtime"]["precision"],
        device,
    )

    set_seed(int(config["project"]["seed"]))
    configure_runtime(config, device)

    print(f"运行设备：{device}")
    print(f"计算精度：{dtype}")

    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["generator_name"]
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}

    if device.type == "cuda" and dtype != torch.float32:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["generator_name"],
        **model_kwargs,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(device)

    if config["model"]["freeze_generator_backbone"]:
        freeze_generator(model)

    if config["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    semantic_encoder = SemanticEncoder(
        model_name=config["model"]["encoder_name"],
        device=device,
        span_size=config["watermark"]["span_size"],
        span_stride=config["watermark"]["span_stride"],
        normalize_embeddings=config["watermark"][
            "normalize_embeddings"
        ],
        freeze=True,
    )

    anchors = AnchorManager(
        secret_key=config["watermark"]["secret_key"],
        num_anchors=config["watermark"]["num_anchors"],
        embedding_dim=semantic_encoder.embedding_dim,
        device=device,
        normalize=config["watermark"][
            "normalize_embeddings"
        ],
    )

    hidden_size = int(model.config.hidden_size)

    projection = SemanticProjection(
        hidden_size=hidden_size,
        embedding_dim=semantic_encoder.embedding_dim,
    ).to(device)

    dataset = WatermarkDataset(
        path=config["training"]["train_file"],
        tokenizer=tokenizer,
        prompt_field=config["data"]["prompt_field"],
        response_field=config["data"]["response_field"],
        max_length=config["model"]["max_input_length"],
    )

    collator = WatermarkCollator(
        pad_token_id=tokenizer.pad_token_id
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["runtime"]["num_workers"],
        pin_memory=(
            config["runtime"]["pin_memory"]
            and device.type == "cuda"
        ),
    )

    loss_fn = WatermarkLoss(
        anchors=anchors,
        lambda_lm=config["training"]["lambda_lm"],
        lambda_semantic=config["training"][
            "lambda_semantic"
        ],
        lambda_persistence=config["training"][
            "lambda_persistence"
        ],
        persistence_noise_std=config["training"][
            "persistence_noise_std"
        ],
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameters.extend(
        projection.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = int(config["training"]["epochs"])
    accumulation_steps = int(
        config["training"]["gradient_accumulation_steps"]
    )

    updates_per_epoch = math.ceil(
        len(dataloader) / accumulation_steps
    )

    total_training_steps = updates_per_epoch * epochs

    warmup_steps = int(
        total_training_steps
        * config["training"]["warmup_ratio"]
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    use_scaler = (
        device.type == "cuda"
        and dtype == torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

    output_dir = Path(
        args.output_dir
        or config["project"]["checkpoint_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    print(f"训练样本数：{len(dataset)}")
    print(f"每轮批次数：{len(dataloader)}")
    print(f"训练轮数：{epochs}")
    print(f"总更新步数：{total_training_steps}")

    for epoch in range(epochs):
        model.train()
        projection.train()

        running_total = 0.0
        running_lm = 0.0
        running_semantic = 0.0
        running_persistence = 0.0

        for batch_index, batch in enumerate(dataloader):
            batch = move_batch(batch, device)

            with get_autocast_context(device, dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                last_hidden_state = outputs.hidden_states[-1]

                semantic_trajectories = (
                    build_semantic_trajectories(
                        hidden_states=last_hidden_state,
                        labels=batch["labels"],
                        projection=projection,
                        span_size=config["watermark"][
                            "span_size"
                        ],
                        span_stride=config["watermark"][
                            "span_stride"
                        ],
                    )
                )

                loss_output = loss_fn.forward_batch(
                    lm_loss=outputs.loss,
                    batch_embeddings=semantic_trajectories,
                )

                scaled_loss = (
                    loss_output.total_loss
                    / accumulation_steps
                )

            scaler.scale(scaled_loss).backward()

            should_update = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(dataloader)
            )

            if should_update:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    config["training"]["max_grad_norm"],
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            loss_values = loss_output.to_dict()

            running_total += loss_values["total_loss"]
            running_lm += loss_values["lm_loss"]
            running_semantic += loss_values["semantic_loss"]
            running_persistence += loss_values[
                "persistence_loss"
            ]

            if (
                batch_index + 1
            ) % config["training"]["log_every"] == 0:
                sample_count = batch_index + 1

                print(
                    f"epoch={epoch + 1}/{epochs} "
                    f"batch={sample_count}/{len(dataloader)} "
                    f"step={global_step} "
                    f"total={running_total / sample_count:.6f} "
                    f"lm={running_lm / sample_count:.6f} "
                    f"semantic="
                    f"{running_semantic / sample_count:.6f} "
                    f"persistence="
                    f"{running_persistence / sample_count:.6f}"
                )

        batch_count = len(dataloader)

        print(
            f"epoch={epoch + 1} 完成 "
            f"total={running_total / batch_count:.6f} "
            f"lm={running_lm / batch_count:.6f} "
            f"semantic={running_semantic / batch_count:.6f} "
            f"persistence="
            f"{running_persistence / batch_count:.6f}"
        )

        if config["training"]["save_each_epoch"]:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                projection=projection,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                output_dir=output_dir,
                config=config,
            )

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    torch.save(
        {
            "state_dict": projection.state_dict(),
            "input_dim": hidden_size,
            "output_dim": semantic_encoder.embedding_dim,
            "global_step": global_step,
        },
        final_dir / "semantic_projection.pt",
    )

    print(f"训练完成，模型保存在：{final_dir}")


if __name__ == "__main__":
    main()