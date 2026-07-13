import argparse
import json
from pathlib import Path

import torch
import yaml

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder
from models.injector import SemanticWatermarkInjector


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

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用的 CUDA GPU")

        return torch.device("cuda")

    if device_name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用的 CUDA GPU")

        gpu_index = int(device_name.split(":", maxsplit=1)[1])

        if gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"请求使用 cuda:{gpu_index}，"
                f"但当前只有 {torch.cuda.device_count()} 张 GPU"
            )

        return torch.device(device_name)

    if device_name == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"不支持的设备配置：{device_name}"
    )


def resolve_dtype(
    precision: str,
    device: torch.device,
) -> torch.dtype | None:
    precision = str(precision).strip().lower()

    if device.type != "cuda":
        return None

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

    raise ValueError(
        f"不支持的精度配置：{precision}"
    )


def configure_runtime(
    config: dict,
    device: torch.device,
) -> None:
    seed = int(config["project"]["seed"])

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if device.type == "cuda":
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


def build_injector(
    config: dict,
    device: torch.device,
    dtype: torch.dtype | None,
) -> SemanticWatermarkInjector:
    encoder = SemanticEncoder(
        model_name=config["model"]["encoder_name"],
        device=device,
        span_size=config["watermark"]["span_size"],
        span_stride=config["watermark"]["span_stride"],
        normalize_embeddings=config["watermark"][
            "normalize_embeddings"
        ],
        freeze=config["model"]["freeze_encoder"],
    )

    anchors = AnchorManager(
        secret_key=config["watermark"]["secret_key"],
        num_anchors=config["watermark"]["num_anchors"],
        embedding_dim=encoder.embedding_dim,
        device=device,
        normalize=config["watermark"][
            "normalize_embeddings"
        ],
    )

    injector = SemanticWatermarkInjector(
        model_name=config["model"]["generator_name"],
        encoder=encoder,
        anchors=anchors,
        device=device,
        num_candidates=config["watermark"][
            "num_candidates"
        ],
        max_input_length=config["model"][
            "max_input_length"
        ],
        max_new_tokens=config["model"][
            "max_new_tokens"
        ],
        quality_weight=config["watermark"][
            "quality_weight"
        ],
        watermark_weight=config["watermark"][
            "watermark_weight"
        ],
        generation_batch_size=config["generation"][
            "batch_size"
        ],
        do_sample=config["generation"]["do_sample"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        top_k=config["generation"]["top_k"],
        repetition_penalty=config["generation"][
            "repetition_penalty"
        ],
        dtype=dtype,
    )

    return injector


def result_to_dict(result) -> dict:
    return {
        "prompt": result.prompt,
        "selected_index": result.selected_index,
        "text": result.text,
        "candidates": [
            {
                "text": candidate.text,
                "quality_score": candidate.quality_score,
                "watermark_score": candidate.watermark_score,
                "normalized_quality_score": (
                    candidate.normalized_quality_score
                ),
                "normalized_watermark_score": (
                    candidate.normalized_watermark_score
                ),
                "final_score": candidate.final_score,
            }
            for candidate in result.candidates
        ],
    }


def save_result(
    result_data: dict,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            result_data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_result(result) -> None:
    print()
    print("=" * 80)
    print("生成完成")
    print("=" * 80)
    print(f"选中候选编号：{result.selected_index}")
    print()
    print("最终文本：")
    print(result.text)
    print()

    print("候选详情：")

    for index, candidate in enumerate(result.candidates):
        selected_mark = " <- selected" if (
            index == result.selected_index
        ) else ""

        print("-" * 80)
        print(f"候选 {index}{selected_mark}")
        print(f"质量分数：{candidate.quality_score:.6f}")
        print(f"水印分数：{candidate.watermark_score:.6f}")
        print(
            "归一化质量分数："
            f"{candidate.normalized_quality_score:.6f}"
        )
        print(
            "归一化水印分数："
            f"{candidate.normalized_watermark_score:.6f}"
        )
        print(f"综合分数：{candidate.final_score:.6f}")
        print("文本：")
        print(candidate.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成语义水印文本"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.device is not None:
        config["project"]["device"] = args.device

    if args.num_candidates is not None:
        if args.num_candidates <= 0:
            raise ValueError(
                "--num-candidates 必须大于 0"
            )

        config["watermark"][
            "num_candidates"
        ] = args.num_candidates

    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise ValueError(
                "--max-new-tokens 必须大于 0"
            )

        config["model"][
            "max_new_tokens"
        ] = args.max_new_tokens

    device = resolve_device(
        config["project"]["device"]
    )

    dtype = resolve_dtype(
        config["runtime"]["precision"],
        device,
    )

    configure_runtime(config, device)

    print(f"运行设备：{device}")

    if device.type == "cuda":
        print(
            "GPU："
            f"{torch.cuda.get_device_name(device)}"
        )
        print(f"计算精度：{dtype}")
    else:
        print("计算精度：float32")

    injector = build_injector(
        config=config,
        device=device,
        dtype=dtype,
    )

    result = injector.generate(args.prompt)

    print_result(result)

    result_data = result_to_dict(result)

    if args.output is not None:
        output_path = args.output
    else:
        output_dir = Path(
            config["project"]["result_dir"]
        )

        output_path = str(
            output_dir / "generation.json"
        )

    save_result(
        result_data=result_data,
        output_path=output_path,
    )

    print()
    print(f"结果已保存到：{output_path}")


if __name__ == "__main__":
    main()