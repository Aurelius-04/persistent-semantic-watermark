from dataclasses import dataclass
import re
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer


@dataclass
class SemanticTrajectory:
    text: str
    spans: list[str]
    embeddings: torch.Tensor

    @property
    def length(self) -> int:
        return len(self.spans)

    @property
    def embedding_dim(self) -> int:
        if self.embeddings.ndim != 2:
            return 0
        return int(self.embeddings.shape[-1])


class SemanticEncoder:
    def __init__(
        self,
        model_name: str,
        device: torch.device | str,
        span_size: int = 12,
        span_stride: int = 8,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        freeze: bool = True,
        min_span_size: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.span_size = span_size
        self.span_stride = span_stride
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.freeze = freeze

        if min_span_size is None:
            min_span_size = max(1, span_size // 2)

        self.min_span_size = min_span_size

        self._validate_config()

        self.model = SentenceTransformer(
            model_name,
            device=str(self.device),
        )

        if self.freeze:
            self._freeze_model()

        self.embedding_dim = int(
            self.model.get_sentence_embedding_dimension()
        )

    def _validate_config(self) -> None:
        if self.span_size <= 0:
            raise ValueError("span_size 必须大于 0")

        if self.span_stride <= 0:
            raise ValueError("span_stride 必须大于 0")

        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if self.min_span_size <= 0:
            raise ValueError("min_span_size 必须大于 0")

        if self.min_span_size > self.span_size:
            raise ValueError("min_span_size 不能大于 span_size")

    def _freeze_model(self) -> None:
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        pattern = re.compile(
            "["
            "\u3400-\u4dbf"
            "\u4e00-\u9fff"
            "\u3040-\u309f"
            "\u30a0-\u30ff"
            "\uac00-\ud7af"
            "]"
        )

        return bool(pattern.search(text))

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")

        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text

    def tokenize_units(self, text: str) -> tuple[list[str], str]:
        text = self._clean_text(text)

        if not text:
            return [], "word"

        if self._contains_cjk(text):
            units = [
                character
                for character in text
                if not character.isspace()
            ]
            return units, "char"

        return text.split(), "word"

    @staticmethod
    def _join_units(
        units: Sequence[str],
        join_mode: str,
    ) -> str:
        if join_mode == "char":
            return "".join(units)

        if join_mode == "word":
            return " ".join(units)

        raise ValueError(f"不支持的 join_mode：{join_mode}")

    def split_spans(self, text: str) -> list[str]:
        units, join_mode = self.tokenize_units(text)

        if not units:
            return []

        if len(units) <= self.span_size:
            return [self._join_units(units, join_mode)]

        span_units: list[list[str]] = []
        start = 0

        while start < len(units):
            end = min(start + self.span_size, len(units))
            current_units = units[start:end]

            if not current_units:
                break

            span_units.append(current_units)

            if end >= len(units):
                break

            start += self.span_stride

        if (
            len(span_units) >= 2
            and len(span_units[-1]) < self.min_span_size
        ):
            previous_span = span_units[-2]
            final_span = span_units[-1]
            merged_span = previous_span + final_span

            if len(merged_span) > self.span_size:
                merged_span = merged_span[-self.span_size:]

            span_units[-2] = merged_span
            span_units.pop()

        return [
            self._join_units(items, join_mode)
            for items in span_units
        ]

    def encode_spans(
        self,
        spans: Sequence[str],
    ) -> torch.Tensor:
        cleaned_spans = [
            self._clean_text(span)
            for span in spans
            if isinstance(span, str) and span.strip()
        ]

        if not cleaned_spans:
            return torch.empty(
                (0, self.embedding_dim),
                dtype=torch.float32,
                device=self.device,
            )

        context = (
            torch.no_grad()
            if self.freeze
            else torch.enable_grad()
        )

        with context:
            embeddings = self.model.encode(
                cleaned_spans,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                device=str(self.device),
            )

        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)

        return embeddings.to(self.device)

    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            return torch.zeros(
                self.embedding_dim,
                dtype=torch.float32,
                device=self.device,
            )

        context = (
            torch.no_grad()
            if self.freeze
            else torch.enable_grad()
        )

        with context:
            embedding = self.model.encode(
                cleaned_text,
                batch_size=1,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                device=str(self.device),
            )

        return embedding.to(self.device)

    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(
                (0, self.embedding_dim),
                dtype=torch.float32,
                device=self.device,
            )

        cleaned_texts = [
            self._clean_text(text)
            for text in texts
        ]

        context = (
            torch.no_grad()
            if self.freeze
            else torch.enable_grad()
        )

        with context:
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                device=str(self.device),
            )

        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)

        return embeddings.to(self.device)

    def build_trajectory(
        self,
        text: str,
    ) -> SemanticTrajectory:
        cleaned_text = self._clean_text(text)
        spans = self.split_spans(cleaned_text)
        embeddings = self.encode_spans(spans)

        return SemanticTrajectory(
            text=cleaned_text,
            spans=spans,
            embeddings=embeddings,
        )

    def mean_pool_trajectory(
        self,
        trajectory: SemanticTrajectory,
    ) -> torch.Tensor:
        if trajectory.embeddings.shape[0] == 0:
            return torch.zeros(
                self.embedding_dim,
                dtype=torch.float32,
                device=self.device,
            )

        pooled = trajectory.embeddings.mean(dim=0)

        if self.normalize_embeddings:
            pooled = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=0,
            )

        return pooled

    def to(
        self,
        device: torch.device | str,
    ) -> "SemanticEncoder":
        self.device = torch.device(device)
        self.model.to(self.device)

        return self

    def train(
        self,
        mode: bool = True,
    ) -> "SemanticEncoder":
        if self.freeze:
            self.model.eval()
        else:
            self.model.train(mode)

        return self

    def eval(self) -> "SemanticEncoder":
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.embedding_dim}, "
            f"span_size={self.span_size}, "
            f"span_stride={self.span_stride}, "
            f"normalize_embeddings={self.normalize_embeddings}, "
            f"device='{self.device}', "
            f"freeze={self.freeze}"
            ")"
        )