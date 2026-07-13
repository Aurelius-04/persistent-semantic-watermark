import torch
import yaml

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder
from models.injector import SemanticWatermarkInjector


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device_name)


def resolve_dtype(
    precision: str,
    device: torch.device,
) -> torch.dtype | None:
    if device.type != "cuda":
        return None

    if precision == "fp16":
        return torch.float16

    if precision == "bf16":
        return torch.bfloat16

    if precision == "fp32":
        return torch.float32

    if precision == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    raise ValueError(f"不支持的 precision：{precision}")


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = resolve_device(
        config["project"]["device"]
    )

    dtype = resolve_dtype(
        config["runtime"]["precision"],
        device,
    )

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

    result = injector.generate(
        "Explain the risks of phishing emails."
    )

    print("设备:", device)
    print("选中候选:", result.selected_index)
    print("生成结果:")
    print(result.text)

    for index, candidate in enumerate(result.candidates):
        print("-" * 60)
        print("候选:", index)
        print("文本:", candidate.text)
        print("质量分数:", candidate.quality_score)
        print("水印分数:", candidate.watermark_score)
        print("综合分数:", candidate.final_score)


if __name__ == "__main__":
    main()