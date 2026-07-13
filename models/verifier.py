from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from models.anchors import AnchorManager
from models.encoder import SemanticEncoder


@dataclass
class TextVerificationResult:
    text: str
    score: float
    threshold: float
    is_watermarked: bool
    span_scores: list[float]
    spans: list[str]


@dataclass
class ProbeVerificationResult:
    prompt: str
    generated_text: str
    score: float


@dataclass
class ModelVerificationResult:
    score: float
    threshold: float
    is_watermarked: bool
    probes: list[ProbeVerificationResult]


@dataclass
class JointVerificationResult:
    text_score: float
    model_score: float
    joint_score: float
    threshold: float
    fusion_alpha: float
    is_watermarked: bool
    text_result: TextVerificationResult
    model_result: ModelVerificationResult


class WatermarkVerifier:
    def __init__(
        self,
        encoder: SemanticEncoder,
        anchors: AnchorManager,
        device: torch.device | str,
        text_threshold: float = 0.2,
        model_threshold: float = 0.2,
        joint_threshold: float = 0.2,
        fusion_alpha: float = 0.7,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_input_length: int = 128,
        max_new_tokens: int = 80,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> None:
        self.encoder = encoder
        self.anchors = anchors
        self.device = torch.device(device)

        self.text_threshold = text_threshold
        self.model_threshold = model_threshold
        self.joint_threshold = joint_threshold
        self.fusion_alpha = fusion_alpha

        self.model = model
        self.tokenizer = tokenizer

        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        self._validate_config()

        self.encoder.to(self.device)
        self.anchors.to(self.device)

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        if self.tokenizer is not None:
            self._prepare_tokenizer()

    def _validate_config(self) -> None:
        if not 0.0 <= self.fusion_alpha <= 1.0:
            raise ValueError("fusion_alpha 必须在 [0, 1] 范围内")

        if self.max_input_length <= 0:
            raise ValueError("max_input_length 必须大于 0")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")

        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p 必须在 (0, 1] 范围内")

        if self.top_k < 0:
            raise ValueError("top_k 不能小于 0")

    def _prepare_tokenizer(self) -> None:
        if self.tokenizer is None:
            return

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "tokenizer 没有 pad_token_id 或 eos_token_id"
                )

            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model is not None:
            self.model.config.pad_token_id = (
                self.tokenizer.pad_token_id
            )

    def set_generator(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self._prepare_tokenizer()

    def verify_text(
        self,
        text: str,
        threshold: float | None = None,
    ) -> TextVerificationResult:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")

        text = text.strip()

        if not text:
            raise ValueError("text 不能为空")

        if threshold is None:
            threshold = self.text_threshold

        trajectory = self.encoder.build_trajectory(text)

        span_score_tensor = self.anchors.similarity_scores(
            trajectory.embeddings
        )

        span_scores = [
            float(score)
            for score in span_score_tensor.detach().cpu().tolist()
        ]

        if span_scores:
            score = sum(span_scores) / len(span_scores)
        else:
            score = 0.0

        return TextVerificationResult(
            text=text,
            score=score,
            threshold=threshold,
            is_watermarked=score >= threshold,
            span_scores=span_scores,
            spans=trajectory.spans,
        )

    def generate_probe_response(
        self,
        prompt: str,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "模型侧验证需要先提供 model 和 tokenizer"
            )

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

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **generation_kwargs
            )

        prompt_length = input_ids.shape[1]
        generated_ids = output_ids[:, prompt_length:]

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        if not generated_text:
            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()

        return generated_text

    def verify_model(
        self,
        probe_prompts: Sequence[str],
        threshold: float | None = None,
        max_probe_samples: int | None = None,
    ) -> ModelVerificationResult:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "模型侧验证需要先提供 model 和 tokenizer"
            )

        if threshold is None:
            threshold = self.model_threshold

        prompts = [
            prompt.strip()
            for prompt in probe_prompts
            if isinstance(prompt, str) and prompt.strip()
        ]

        if max_probe_samples is not None:
            if max_probe_samples <= 0:
                raise ValueError(
                    "max_probe_samples 必须大于 0"
                )

            prompts = prompts[:max_probe_samples]

        if not prompts:
            raise ValueError("probe_prompts 不能为空")

        probe_results: list[ProbeVerificationResult] = []

        for prompt in prompts:
            generated_text = self.generate_probe_response(prompt)
            text_result = self.verify_text(
                generated_text,
                threshold=self.text_threshold,
            )

            probe_results.append(
                ProbeVerificationResult(
                    prompt=prompt,
                    generated_text=generated_text,
                    score=text_result.score,
                )
            )

        model_score = sum(
            result.score
            for result in probe_results
        ) / len(probe_results)

        return ModelVerificationResult(
            score=model_score,
            threshold=threshold,
            is_watermarked=model_score >= threshold,
            probes=probe_results,
        )

    def verify_joint(
        self,
        text: str,
        probe_prompts: Sequence[str],
        threshold: float | None = None,
        fusion_alpha: float | None = None,
        max_probe_samples: int | None = None,
    ) -> JointVerificationResult:
        if threshold is None:
            threshold = self.joint_threshold

        if fusion_alpha is None:
            fusion_alpha = self.fusion_alpha

        if not 0.0 <= fusion_alpha <= 1.0:
            raise ValueError(
                "fusion_alpha 必须在 [0, 1] 范围内"
            )

        text_result = self.verify_text(
            text,
            threshold=self.text_threshold,
        )

        model_result = self.verify_model(
            probe_prompts=probe_prompts,
            threshold=self.model_threshold,
            max_probe_samples=max_probe_samples,
        )

        joint_score = (
            fusion_alpha * text_result.score
            + (1.0 - fusion_alpha) * model_result.score
        )

        return JointVerificationResult(
            text_score=text_result.score,
            model_score=model_result.score,
            joint_score=joint_score,
            threshold=threshold,
            fusion_alpha=fusion_alpha,
            is_watermarked=joint_score >= threshold,
            text_result=text_result,
            model_result=model_result,
        )

    @staticmethod
    def load_probe_prompts(
        path: str | Path,
        prompt_field: str = "prompt",
        max_samples: int | None = None,
    ) -> list[str]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"探针文件不存在：{path}"
            )

        prompts: list[str] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path} 第 {line_number} 行不是有效 JSON"
                    ) from error

                prompt = item.get(prompt_field)

                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"{path} 第 {line_number} 行缺少有效的 "
                        f"{prompt_field} 字段"
                    )

                prompts.append(prompt.strip())

                if (
                    max_samples is not None
                    and len(prompts) >= max_samples
                ):
                    break

        if not prompts:
            raise ValueError(
                f"探针文件中没有有效 prompt：{path}"
            )

        return prompts

    def to(
        self,
        device: torch.device | str,
    ) -> "WatermarkVerifier":
        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.anchors.to(self.device)

        if self.model is not None:
            self.model.to(self.device)

        return self

    def eval(self) -> "WatermarkVerifier":
        self.encoder.eval()

        if self.model is not None:
            self.model.eval()

        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"text_threshold={self.text_threshold}, "
            f"model_threshold={self.model_threshold}, "
            f"joint_threshold={self.joint_threshold}, "
            f"fusion_alpha={self.fusion_alpha}, "
            f"device='{self.device}'"
            ")"
        )