import torch
import yaml

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder
from models.injector import SemanticWatermarkInjector
from models.verifier import WatermarkVerifier


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(name)


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
        num_candidates=config["watermark"]["num_candidates"],
        max_input_length=config["model"]["max_input_length"],
        max_new_tokens=config["model"]["max_new_tokens"],
        quality_weight=config["watermark"]["quality_weight"],
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
        model=injector.model,
        tokenizer=injector.tokenizer,
        max_input_length=config["model"][
            "max_input_length"
        ],
        max_new_tokens=config["model"]["max_new_tokens"],
        do_sample=config["generation"]["do_sample"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        top_k=config["generation"]["top_k"],
        repetition_penalty=config["generation"][
            "repetition_penalty"
        ],
    )

    generation = injector.generate(
        "Explain the risks of phishing emails."
    )

    probes = verifier.load_probe_prompts(
        config["verification"]["probe_file"],
        prompt_field=config["data"]["prompt_field"],
        max_samples=config["verification"][
            "max_probe_samples"
        ],
    )

    text_result = verifier.verify_text(
        generation.text
    )

    print("生成文本:")
    print(generation.text)
    print()
    print("文本侧分数:", text_result.score)
    print("文本侧判断:", text_result.is_watermarked)

    model_result = verifier.verify_model(
        probe_prompts=probes,
        max_probe_samples=3,
    )

    print()
    print("模型侧分数:", model_result.score)
    print("模型侧判断:", model_result.is_watermarked)

    joint_result = verifier.verify_joint(
        text=generation.text,
        probe_prompts=probes,
        max_probe_samples=3,
    )

    print()
    print("联合分数:", joint_result.joint_score)
    print("联合判断:", joint_result.is_watermarked)

    for probe in model_result.probes:
        print("-" * 60)
        print("探针:", probe.prompt)
        print("输出:", probe.generated_text)
        print("分数:", probe.score)


if __name__ == "__main__":
    main()