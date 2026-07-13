import torch

from models.anchors import AnchorManager
from models.losses import WatermarkLoss


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    anchors = AnchorManager(
        secret_key=2026,
        num_anchors=8,
        embedding_dim=384,
        device=device,
    )

    loss_fn = WatermarkLoss(
        anchors=anchors,
        lambda_lm=1.0,
        lambda_semantic=0.1,
        lambda_persistence=0.05,
        persistence_noise_std=0.02,
    )

    lm_loss = torch.tensor(
        2.5,
        device=device,
        requires_grad=True,
    )

    embeddings = torch.randn(
        4,
        384,
        device=device,
        requires_grad=True,
    )

    output = loss_fn(
        lm_loss=lm_loss,
        semantic_embeddings=embeddings,
    )

    output.total_loss.backward()

    print(output.to_dict())
    print("embeddings grad:", embeddings.grad is not None)
    print("lm loss grad:", lm_loss.grad)


if __name__ == "__main__":
    main()