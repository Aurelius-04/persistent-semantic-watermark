import hashlib

import torch
import torch.nn.functional as F


class AnchorManager:
    def __init__(
        self,
        secret_key: int | str,
        num_anchors: int,
        embedding_dim: int,
        device: torch.device | str,
        normalize: bool = True,
    ) -> None:
        self.secret_key = str(secret_key)
        self.num_anchors = num_anchors
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.normalize = normalize

        self._validate_config()
        self.seed = self._key_to_seed(self.secret_key)
        self.anchors = self._build_anchors()

    def _validate_config(self) -> None:
        if self.num_anchors <= 0:
            raise ValueError("num_anchors 必须大于 0")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")

    @staticmethod
    def _key_to_seed(secret_key: str) -> int:
        digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") % (2**63 - 1)

    def _build_anchors(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        anchors = torch.randn(
            self.num_anchors,
            self.embedding_dim,
            generator=generator,
            dtype=torch.float32,
        )

        if self.normalize:
            anchors = F.normalize(
                anchors,
                p=2,
                dim=-1,
            )

        return anchors.to(self.device)

    def get_anchor_index(self, position: int) -> int:
        if position < 0:
            raise ValueError("position 不能小于 0")

        position_key = f"{self.secret_key}:{position}"
        digest = hashlib.sha256(
            position_key.encode("utf-8")
        ).digest()

        value = int.from_bytes(
            digest[:8],
            byteorder="big",
        )

        return value % self.num_anchors

    def get_anchor(self, position: int) -> torch.Tensor:
        index = self.get_anchor_index(position)
        return self.anchors[index]

    def get_indices(
        self,
        length: int,
    ) -> list[int]:
        if length < 0:
            raise ValueError("length 不能小于 0")

        return [
            self.get_anchor_index(position)
            for position in range(length)
        ]

    def get_trajectory(
        self,
        length: int,
    ) -> torch.Tensor:
        if length < 0:
            raise ValueError("length 不能小于 0")

        if length == 0:
            return torch.empty(
                (0, self.embedding_dim),
                dtype=self.anchors.dtype,
                device=self.device,
            )

        indices = torch.tensor(
            self.get_indices(length),
            dtype=torch.long,
            device=self.device,
        )

        return self.anchors.index_select(
            dim=0,
            index=indices,
        )

    def similarity_scores(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings 必须是二维 Tensor，"
                "形状应为 [num_spans, embedding_dim]"
            )

        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"embedding_dim 不匹配："
                f"期望 {self.embedding_dim}，"
                f"实际 {embeddings.shape[-1]}"
            )

        if embeddings.shape[0] == 0:
            return torch.empty(
                0,
                dtype=torch.float32,
                device=self.device,
            )

        embeddings = embeddings.to(self.device)
        anchors = self.get_trajectory(embeddings.shape[0])

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1,
        )

        anchors = F.normalize(
            anchors,
            p=2,
            dim=-1,
        )

        return F.cosine_similarity(
            embeddings,
            anchors,
            dim=-1,
        )

    def score(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.similarity_scores(embeddings)

        if scores.numel() == 0:
            return torch.tensor(
                0.0,
                dtype=torch.float32,
                device=self.device,
            )

        return scores.mean()

    def alignment_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return 1.0 - self.score(embeddings)

    def regenerate(
        self,
        secret_key: int | str | None = None,
    ) -> None:
        if secret_key is not None:
            self.secret_key = str(secret_key)

        self.seed = self._key_to_seed(self.secret_key)
        self.anchors = self._build_anchors()

    def to(
        self,
        device: torch.device | str,
    ) -> "AnchorManager":
        self.device = torch.device(device)
        self.anchors = self.anchors.to(self.device)

        return self

    def state_dict(self) -> dict:
        return {
            "secret_key": self.secret_key,
            "num_anchors": self.num_anchors,
            "embedding_dim": self.embedding_dim,
            "normalize": self.normalize,
            "anchors": self.anchors.detach().cpu(),
        }

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        required_keys = {
            "secret_key",
            "num_anchors",
            "embedding_dim",
            "normalize",
            "anchors",
        }

        missing_keys = required_keys - set(state_dict)

        if missing_keys:
            raise KeyError(
                f"state_dict 缺少字段：{sorted(missing_keys)}"
            )

        self.secret_key = str(state_dict["secret_key"])
        self.num_anchors = int(state_dict["num_anchors"])
        self.embedding_dim = int(state_dict["embedding_dim"])
        self.normalize = bool(state_dict["normalize"])
        self.seed = self._key_to_seed(self.secret_key)

        anchors = state_dict["anchors"]

        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(
                anchors,
                dtype=torch.float32,
            )

        expected_shape = (
            self.num_anchors,
            self.embedding_dim,
        )

        if tuple(anchors.shape) != expected_shape:
            raise ValueError(
                f"anchors 形状不匹配："
                f"期望 {expected_shape}，"
                f"实际 {tuple(anchors.shape)}"
            )

        self.anchors = anchors.to(self.device)

    def __len__(self) -> int:
        return self.num_anchors

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_anchors={self.num_anchors}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device='{self.device}', "
            f"normalize={self.normalize}"
            ")"
        )