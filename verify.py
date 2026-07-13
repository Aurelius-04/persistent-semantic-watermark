import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder
from models.verifier import WatermarkVerifier


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
            raise RuntimeError("当前环境没有可用的 CUDA GPU")

        return torch.device("cuda")

    if device_name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用的 CUDA GPU")

        try:
            gpu_index = int(
                device_name.split(":", maxsplit=1)[1]
            )
        except ValueError as error:
            raise ValueError(
                f"无效的设备配置：{device_name}"
            ) from error

        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"请求使用 cuda:{gpu_index}，"
                f"但当前只有 {torch.cuda.device_count()} 张 GPU"
            )

        return torch.device(device_name)

    raise ValueError(f"不支持的设备配置：{device_name}")


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

    raise ValueError(f"不支持的精度配置：{precision}")


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


def load_generator(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype | None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}

    if device.type == "cuda" and dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    model.to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def build_verifier(
    config: dict,
    device: torch.device,
    dtype: torch.dtype | None,
    model_path: str | None = None,
) -> WatermarkVerifier:
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

    model = None
    tokenizer = None

    if model_path is not None:
        model, tokenizer = load_generator(
            model_name_or_path=model_path,
            device=device,
            dtype=dtype,
        )

    verifier = WatermarkVerifier(
        encoder=encoder,
        anchors=anchors,
        device=device,
        text_threshold=config["verification"][
            "text_threshold"
        ],
        model_threshold=config["verification"][
            "model_threshold"
        ],
        joint_threshold=config["verification"][
            "joint_threshold"
        ],
        fusion_alpha=config["verification"][
            "fusion_alpha"
        ],
        model=model,
        tokenizer=tokenizer,
        max_input_length=config["model"][
            "max_input_length"
        ],
        max_new_tokens=config["model"][
            "max_new_tokens"
        ],
        do_sample=config["generation"]["do_sample"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        top_k=config["generation"]["top_k"],
        repetition_penalty=config["generation"][
            "repetition_penalty"
        ],
    )

    return verifier


def load_text_from_file(path: str) -> str:
    text_path = Path(path)

    if not text_path.exists():
        raise FileNotFoundError(
            f"文本文件不存在：{text_path}"
        )

    text = text_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError("文本文件内容为空")

    return text


def text_result_to_dict(result) -> dict:
    return {
        "mode": "text",
        "text": result.text,
        "score": result.score,
        "threshold": result.threshold,
        "is_watermarked": result.is_watermarked,
        "spans": result.spans,
        "span_scores": result.span_scores,
    }


def model_result_to_dict(result) -> dict:
    return {
        "mode": "model",
        "score": result.score,
        "threshold": result.threshold,
        "is_watermarked": result.is_watermarked,
        "probes": [
            {
                "prompt": probe.prompt,
                "generated_text": probe.generated_text,
                "score": probe.score,
            }
            for probe in result.probes
        ],
    }


def joint_result_to_dict(result) -> dict:
    return {
        "mode": "joint",
        "text_score": result.text_score,
        "model_score": result.model_score,
        "joint_score": result.joint_score,
        "threshold": result.threshold,
        "fusion_alpha": result.fusion_alpha,
        "is_watermarked": result.is_watermarked,
        "text_result": text_result_to_dict(
            result.text_result
        ),
        "model_result": model_result_to_dict(
            result.model_result
        ),
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


def print_text_result(result) -> None:
    print()
    print("=" * 80)
    print("文本侧验证结果")
    print("=" * 80)
    print(f"分数：{result.score:.6f}")
    print(f"阈值：{result.threshold:.6f}")
    print(
        "判断："
        + (
            "WATERMARKED"
            if result.is_watermarked
            else "NOT WATERMARKED"
        )
    )

    print()
    print("片段详情：")

    for index, (span, score) in enumerate(
        zip(result.spans, result.span_scores)
    ):
        print("-" * 80)
        print(f"片段 {index}")
        print(f"分数：{score:.6f}")
        print(span)


def print_model_result(result) -> None:
    print()
    print("=" * 80)
    print("模型侧验证结果")
    print("=" * 80)
    print(f"分数：{result.score:.6f}")
    print(f"阈值：{result.threshold:.6f}")
    print(
        "判断："
        + (
            "WATERMARKED"
            if result.is_watermarked
            else "NOT WATERMARKED"
        )
    )

    print()
    print("探针详情：")

    for index, probe in enumerate(result.probes):
        print("-" * 80)
        print(f"探针 {index}")
        print(f"Prompt：{probe.prompt}")
        print(f"分数：{probe.score:.6f}")
        print("生成文本：")
        print(probe.generated_text)


def print_joint_result(result) -> None:
    print()
    print("=" * 80)
    print("联合验证结果")
    print("=" * 80)
    print(f"文本侧分数：{result.text_score:.6f}")
    print(f"模型侧分数：{result.model_score:.6f}")
    print(f"联合分数：{result.joint_score:.6f}")
    print(f"融合权重：{result.fusion_alpha:.6f}")
    print(f"阈值：{result.threshold:.6f}")
    print(
        "判断："
        + (
            "WATERMARKED"
            if result.is_watermarked
            else "NOT WATERMARKED"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="语义水印验证工具"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["text", "model", "joint"],
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--probe-file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.device is not None:
        config["project"]["device"] = args.device

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

    model_path = args.model_path

    if args.mode in {"model", "joint"}:
        if model_path is None:
            model_path = config["model"]["generator_name"]

    verifier = build_verifier(
        config=config,
        device=device,
        dtype=dtype,
        model_path=model_path,
    )

    text = None

    if args.text is not None:
        text = args.text.strip()

    if args.text_file is not None:
        text = load_text_from_file(args.text_file)

    if args.mode in {"text", "joint"} and not text:
        raise ValueError(
            "text 或 joint 模式必须提供 "
            "--text 或 --text-file"
        )

    if args.mode == "text":
        result = verifier.verify_text(
            text=text,
            threshold=args.threshold,
        )

        result_data = text_result_to_dict(result)
        print_text_result(result)

    elif args.mode == "model":
        probe_file = (
            args.probe_file
            or config["verification"]["probe_file"]
        )

        max_probes = (
            args.max_probes
            if args.max_probes is not None
            else config["verification"][
                "max_probe_samples"
            ]
        )

        probes = verifier.load_probe_prompts(
            path=probe_file,
            prompt_field=config["data"]["prompt_field"],
            max_samples=max_probes,
        )

        result = verifier.verify_model(
            probe_prompts=probes,
            threshold=args.threshold,
            max_probe_samples=max_probes,
        )

        result_data = model_result_to_dict(result)
        print_model_result(result)

    else:
        probe_file = (
            args.probe_file
            or config["verification"]["probe_file"]
        )

        max_probes = (
            args.max_probes
            if args.max_probes is not None
            else config["verification"][
                "max_probe_samples"
            ]
        )

        probes = verifier.load_probe_prompts(
            path=probe_file,
            prompt_field=config["data"]["prompt_field"],
            max_samples=max_probes,
        )

        result = verifier.verify_joint(
            text=text,
            probe_prompts=probes,
            threshold=args.threshold,
            fusion_alpha=args.fusion_alpha,
            max_probe_samples=max_probes,
        )

        result_data = joint_result_to_dict(result)
        print_joint_result(result)

    if args.output is not None:
        output_path = args.output
    else:
        output_dir = Path(
            config["project"]["result_dir"]
        )

        output_path = str(
            output_dir / f"verification_{args.mode}.json"
        )

    save_result(
        result_data=result_data,
        output_path=output_path,
    )

    print()
    print(f"结果已保存到：{output_path}")


if __name__ == "__main__":
    main()