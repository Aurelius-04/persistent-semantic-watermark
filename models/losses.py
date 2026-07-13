from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.anchors import AnchorManager


@dataclass
class WatermarkLossOutput:
    total_loss: torch.Tensor
    lm_loss: torch.Tensor
    semantic_loss: torch.Tensor
    persistence_loss: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total_loss.detach().cpu()),
            "lm_loss": float(self.lm_loss.detach().cpu()),
            "semantic_loss": float(
                self.semantic_loss.detach().cpu()
            ),
            "persistence_loss": float(
                self.persistence_loss.detach().cpu()
            ),
        }


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            "logits 必须是三维 Tensor，"
            "形状为 [batch_size, sequence_length, vocab_size]"
        )

    if labels.ndim != 2:
        raise ValueError(
            "labels 必须是二维 Tensor，"
            "形状为 [batch_size, sequence_length]"
        )

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits 和 labels 的前两维不匹配："
            f"{tuple(logits.shape[:2])} 与 {tuple(labels.shape)}"
        )

    if logits.shape[1] < 2:
        return logits.sum() * 0.0

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.shape[-1],
        ),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def semantic_alignment_loss(
    embeddings: torch.Tensor,
    target_anchors: torch.Tensor,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings 必须是二维 Tensor，"
            "形状为 [num_spans, embedding_dim]"
        )

    if target_anchors.ndim != 2:
        raise ValueError(
            "target_anchors 必须是二维 Tensor，"
            "形状为 [num_spans, embedding_dim]"
        )

    if embeddings.shape != target_anchors.shape:
        raise ValueError(
            f"embeddings 与 target_anchors 形状不匹配："
            f"{tuple(embeddings.shape)} 与 "
            f"{tuple(target_anchors.shape)}"
        )

    if embeddings.shape[0] == 0:
        return embeddings.sum() * 0.0

    embeddings = F.normalize(
        embeddings.float(),
        p=2,
        dim=-1,
    )

    target_anchors = F.normalize(
        target_anchors.to(
            device=embeddings.device,
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )

    similarities = F.cosine_similarity(
        embeddings,
        target_anchors,
        dim=-1,
    )

    return (1.0 - similarities).mean()


def anchor_alignment_loss(
    embeddings: torch.Tensor,
    anchors: AnchorManager,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings 必须是二维 Tensor，"
            "形状为 [num_spans, embedding_dim]"
        )

    if embeddings.shape[-1] != anchors.embedding_dim:
        raise ValueError(
            f"嵌入维度不匹配：期望 {anchors.embedding_dim}，"
            f"实际 {embeddings.shape[-1]}"
        )

    target_anchors = anchors.get_trajectory(
        embeddings.shape[0]
    ).to(
        device=embeddings.device,
        dtype=embeddings.dtype,
    )

    return semantic_alignment_loss(
        embeddings,
        target_anchors,
    )


def batch_anchor_alignment_loss(
    batch_embeddings: list[torch.Tensor],
    anchors: AnchorManager,
) -> torch.Tensor:
    if not batch_embeddings:
        return torch.tensor(
            0.0,
            dtype=torch.float32,
            device=anchors.device,
        )

    losses = [
        anchor_alignment_loss(
            embeddings,
            anchors,
        )
        for embeddings in batch_embeddings
    ]

    if not losses:
        return torch.tensor(
            0.0,
            dtype=torch.float32,
            device=anchors.device,
        )

    return torch.stack(losses).mean()


def persistence_consistency_loss(
    original_embeddings: torch.Tensor,
    transformed_embeddings: torch.Tensor,
) -> torch.Tensor:
    if original_embeddings.ndim != 2:
        raise ValueError(
            "original_embeddings 必须是二维 Tensor"
        )

    if transformed_embeddings.ndim != 2:
        raise ValueError(
            "transformed_embeddings 必须是二维 Tensor"
        )

    if original_embeddings.shape != transformed_embeddings.shape:
        raise ValueError(
            "original_embeddings 与 transformed_embeddings "
            "形状必须一致"
        )

    if original_embeddings.shape[0] == 0:
        return original_embeddings.sum() * 0.0

    original_embeddings = F.normalize(
        original_embeddings.float(),
        p=2,
        dim=-1,
    )

    transformed_embeddings = F.normalize(
        transformed_embeddings.float(),
        p=2,
        dim=-1,
    )

    cosine_loss = (
        1.0
        - F.cosine_similarity(
            original_embeddings,
            transformed_embeddings,
            dim=-1,
        )
    ).mean()

    original_mean = original_embeddings.mean(dim=0)
    transformed_mean = transformed_embeddings.mean(dim=0)

    trajectory_loss = F.mse_loss(
        transformed_mean,
        original_mean,
    )

    return cosine_loss + trajectory_loss


def watermark_score_consistency_loss(
    original_embeddings: torch.Tensor,
    transformed_embeddings: torch.Tensor,
    anchors: AnchorManager,
) -> torch.Tensor:
    if original_embeddings.shape != transformed_embeddings.shape:
        raise ValueError(
            "original_embeddings 与 transformed_embeddings "
            "形状必须一致"
        )

    if original_embeddings.shape[0] == 0:
        return original_embeddings.sum() * 0.0

    original_scores = anchors.similarity_scores(
        original_embeddings
    )

    transformed_scores = anchors.similarity_scores(
        transformed_embeddings
    )

    original_scores = original_scores.to(
        transformed_scores.device
    )

    return F.mse_loss(
        transformed_scores.float(),
        original_scores.float(),
    )


def add_embedding_noise(
    embeddings: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    if noise_std < 0:
        raise ValueError("noise_std 不能小于 0")

    if noise_std == 0 or embeddings.numel() == 0:
        return embeddings

    noise = torch.randn_like(embeddings) * noise_std

    return embeddings + noise


class WatermarkLoss(nn.Module):
    def __init__(
        self,
        anchors: AnchorManager,
        lambda_lm: float = 1.0,
        lambda_semantic: float = 0.1,
        lambda_persistence: float = 0.05,
        persistence_noise_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.anchors = anchors
        self.lambda_lm = lambda_lm
        self.lambda_semantic = lambda_semantic
        self.lambda_persistence = lambda_persistence
        self.persistence_noise_std = persistence_noise_std

        self._validate_config()

    def _validate_config(self) -> None:
        if self.lambda_lm < 0:
            raise ValueError("lambda_lm 不能小于 0")

        if self.lambda_semantic < 0:
            raise ValueError("lambda_semantic 不能小于 0")

        if self.lambda_persistence < 0:
            raise ValueError("lambda_persistence 不能小于 0")

        if self.persistence_noise_std < 0:
            raise ValueError(
                "persistence_noise_std 不能小于 0"
            )

        if (
            self.lambda_lm
            + self.lambda_semantic
            + self.lambda_persistence
            <= 0
        ):
            raise ValueError("至少有一个损失权重必须大于 0")

    def forward(
        self,
        lm_loss: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor | None = None,
    ) -> WatermarkLossOutput:
        if lm_loss.ndim != 0:
            lm_loss = lm_loss.mean()

        semantic_loss = anchor_alignment_loss(
            semantic_embeddings,
            self.anchors,
        )

        if transformed_embeddings is None:
            transformed_embeddings = add_embedding_noise(
                semantic_embeddings,
                self.persistence_noise_std,
            )

        embedding_persistence_loss = (
            persistence_consistency_loss(
                semantic_embeddings,
                transformed_embeddings,
            )
        )

        score_persistence_loss = (
            watermark_score_consistency_loss(
                semantic_embeddings,
                transformed_embeddings,
                self.anchors,
            )
        )

        persistence_loss = (
            embedding_persistence_loss
            + score_persistence_loss
        )

        total_loss = (
            self.lambda_lm * lm_loss
            + self.lambda_semantic * semantic_loss
            + self.lambda_persistence * persistence_loss
        )

        return WatermarkLossOutput(
            total_loss=total_loss,
            lm_loss=lm_loss,
            semantic_loss=semantic_loss,
            persistence_loss=persistence_loss,
        )

    def forward_batch(
        self,
        lm_loss: torch.Tensor,
        batch_embeddings: list[torch.Tensor],
        transformed_batch_embeddings: (
            list[torch.Tensor] | None
        ) = None,
    ) -> WatermarkLossOutput:
        if not batch_embeddings:
            zero = lm_loss * 0.0

            total_loss = self.lambda_lm * lm_loss

            return WatermarkLossOutput(
                total_loss=total_loss,
                lm_loss=lm_loss,
                semantic_loss=zero,
                persistence_loss=zero,
            )

        semantic_losses = []
        persistence_losses = []

        for index, embeddings in enumerate(batch_embeddings):
            semantic_losses.append(
                anchor_alignment_loss(
                    embeddings,
                    self.anchors,
                )
            )

            if transformed_batch_embeddings is None:
                transformed = add_embedding_noise(
                    embeddings,
                    self.persistence_noise_std,
                )
            else:
                if len(transformed_batch_embeddings) != len(
                    batch_embeddings
                ):
                    raise ValueError(
                        "transformed_batch_embeddings 与 "
                        "batch_embeddings 长度必须一致"
                    )

                transformed = transformed_batch_embeddings[index]

            embedding_loss = persistence_consistency_loss(
                embeddings,
                transformed,
            )

            score_loss = watermark_score_consistency_loss(
                embeddings,
                transformed,
                self.anchors,
            )

            persistence_losses.append(
                embedding_loss + score_loss
            )

        semantic_loss = torch.stack(
            semantic_losses
        ).mean()

        persistence_loss = torch.stack(
            persistence_losses
        ).mean()

        total_loss = (
            self.lambda_lm * lm_loss
            + self.lambda_semantic * semantic_loss
            + self.lambda_persistence * persistence_loss
        )

        return WatermarkLossOutput(
            total_loss=total_loss,
            lm_loss=lm_loss,
            semantic_loss=semantic_loss,
            persistence_loss=persistence_loss,
        )

    def extra_repr(self) -> str:
        return (
            f"lambda_lm={self.lambda_lm}, "
            f"lambda_semantic={self.lambda_semantic}, "
            f"lambda_persistence={self.lambda_persistence}, "
            f"persistence_noise_std="
            f"{self.persistence_noise_std}"
        )