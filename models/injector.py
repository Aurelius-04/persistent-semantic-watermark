from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder


@dataclass
class CandidateResult:
    text: str
    quality_score: float
    watermark_score: float
    normalized_quality_score: float
    normalized_watermark_score: float
    final_score: float


@dataclass
class WatermarkedGeneration:
    prompt: str
    text: str
    selected_index: int
    candidates: list[CandidateResult]


class SemanticWatermarkInjector:
    def __init__(
        self,
        model_name: str,
        encoder: SemanticEncoder,
        anchors: AnchorManager,
        device: torch.device | str,
        num_candidates: int = 4,
        max_input_length: int = 128,
        max_new_tokens: int = 80,
        quality_weight: float = 0.7,
        watermark_weight: float = 0.3,
        generation_batch_size: int = 1,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.encoder = encoder
        self.anchors = anchors
        self.num_candidates = num_candidates
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.quality_weight = quality_weight
        self.watermark_weight = watermark_weight
        self.generation_batch_size = generation_batch_size
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.dtype = dtype

        self._validate_config()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}

        if self.device.type == "cuda" and self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        self.model.to(self.device)
        self.model.eval()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _validate_config(self) -> None:
        if self.num_candidates <= 0:
            raise ValueError("num_candidates 必须大于 0")

        if self.max_input_length <= 0:
            raise ValueError("max_input_length 必须大于 0")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")

        if self.generation_batch_size <= 0:
            raise ValueError("generation_batch_size 必须大于 0")

        if self.quality_weight < 0:
            raise ValueError("quality_weight 不能小于 0")

        if self.watermark_weight < 0:
            raise ValueError("watermark_weight 不能小于 0")

        if self.quality_weight + self.watermark_weight <= 0:
            raise ValueError("quality_weight 和 watermark_weight 不能同时为 0")

        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")

        if self.top_k < 0:
            raise ValueError("top_k 不能小于 0")

    def generate_candidates(
        self,
        prompt: str,
    ) -> list[str]:
        if not isinstance(prompt, str):
            raise TypeError("prompt 必须是字符串")

        prompt = prompt.strip()

        if not prompt:
            raise ValueError("prompt 不能为空")

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        generated_texts: list[str] = []
        remaining = self.num_candidates

        while remaining > 0:
            current_batch_size = min(
                self.generation_batch_size,
                remaining,
            )

            batch_input_ids = input_ids.repeat(
                current_batch_size,
                1,
            )

            batch_attention_mask = attention_mask.repeat(
                current_batch_size,
                1,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_length = batch_input_ids.shape[1]
            generated_ids = outputs[:, prompt_length:]

            texts = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            for text in texts:
                cleaned_text = text.strip()

                if cleaned_text:
                    generated_texts.append(cleaned_text)
                else:
                    generated_texts.append(prompt)

            remaining -= current_batch_size

        return generated_texts[: self.num_candidates]

    def quality_score(
        self,
        prompt: str,
        candidate: str,
    ) -> float:
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
            add_special_tokens=False,
        )["input_ids"]

        candidate_ids = self.tokenizer(
            candidate,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_new_tokens,
            add_special_tokens=False,
        )["input_ids"]

        if candidate_ids.shape[1] == 0:
            return float("-inf")

        input_ids = torch.cat(
            [prompt_ids, candidate_ids],
            dim=1,
        ).to(self.device)

        attention_mask = torch.ones_like(
            input_ids,
            device=self.device,
        )

        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return -float(outputs.loss.item())

    def watermark_score(
        self,
        candidate: str,
    ) -> float:
        trajectory = self.encoder.build_trajectory(candidate)

        score = self.anchors.score(
            trajectory.embeddings
        )

        return float(score.item())

    @staticmethod
    def _min_max_normalize(
        values: list[float],
    ) -> list[float]:
        if not values:
            return []

        finite_values = [
            value
            for value in values
            if value != float("inf")
            and value != float("-inf")
        ]

        if not finite_values:
            return [0.0 for _ in values]

        minimum = min(finite_values)
        maximum = max(finite_values)

        if maximum - minimum < 1e-8:
            return [
                1.0 if value in finite_values else 0.0
                for value in values
            ]

        normalized = []

        for value in values:
            if value == float("inf"):
                normalized.append(1.0)
            elif value == float("-inf"):
                normalized.append(0.0)
            else:
                normalized.append(
                    (value - minimum) / (maximum - minimum)
                )

        return normalized

    def score_candidates(
        self,
        prompt: str,
        candidates: list[str],
    ) -> list[CandidateResult]:
        quality_scores = [
            self.quality_score(prompt, candidate)
            for candidate in candidates
        ]

        watermark_scores = [
            self.watermark_score(candidate)
            for candidate in candidates
        ]

        normalized_quality_scores = self._min_max_normalize(
            quality_scores
        )

        normalized_watermark_scores = self._min_max_normalize(
            watermark_scores
        )

        total_weight = (
            self.quality_weight
            + self.watermark_weight
        )

        quality_weight = self.quality_weight / total_weight
        watermark_weight = self.watermark_weight / total_weight

        results = []

        for index, candidate in enumerate(candidates):
            final_score = (
                quality_weight
                * normalized_quality_scores[index]
                + watermark_weight
                * normalized_watermark_scores[index]
            )

            results.append(
                CandidateResult(
                    text=candidate,
                    quality_score=quality_scores[index],
                    watermark_score=watermark_scores[index],
                    normalized_quality_score=(
                        normalized_quality_scores[index]
                    ),
                    normalized_watermark_score=(
                        normalized_watermark_scores[index]
                    ),
                    final_score=final_score,
                )
            )

        return results

    def generate(
        self,
        prompt: str,
    ) -> WatermarkedGeneration:
        candidates = self.generate_candidates(prompt)
        scored_candidates = self.score_candidates(
            prompt,
            candidates,
        )

        selected_index = max(
            range(len(scored_candidates)),
            key=lambda index: scored_candidates[index].final_score,
        )

        selected = scored_candidates[selected_index]

        return WatermarkedGeneration(
            prompt=prompt,
            text=selected.text,
            selected_index=selected_index,
            candidates=scored_candidates,
        )

    def to(
        self,
        device: torch.device | str,
    ) -> "SemanticWatermarkInjector":
        self.device = torch.device(device)
        self.model.to(self.device)
        self.encoder.to(self.device)
        self.anchors.to(self.device)

        return self

    def eval(self) -> "SemanticWatermarkInjector":
        self.model.eval()
        self.encoder.eval()

        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_candidates={self.num_candidates}, "
            f"max_input_length={self.max_input_length}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"quality_weight={self.quality_weight}, "
            f"watermark_weight={self.watermark_weight}, "
            f"device='{self.device}'"
            ")"
        )