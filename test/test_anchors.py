import yaml

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder


def resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return name


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = resolve_device(config["project"]["device"])

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

    text = (
        "Phishing emails can steal personal information "
        "by directing users to fake login pages."
    )

    trajectory = encoder.build_trajectory(text)

    scores = anchors.similarity_scores(
        trajectory.embeddings
    )

    overall_score = anchors.score(
        trajectory.embeddings
    )

    loss = anchors.alignment_loss(
        trajectory.embeddings
    )

    print(encoder)
    print(anchors)
    print("轨迹形状:", trajectory.embeddings.shape)
    print("锚点索引:", anchors.get_indices(trajectory.length))
    print("片段分数:", scores)
    print("总体分数:", overall_score.item())
    print("对齐损失:", loss.item())


if __name__ == "__main__":
    main()