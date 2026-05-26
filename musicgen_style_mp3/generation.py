#!/usr/bin/env python3
"""MusicGen-Style 模型适配层与串行分段生成。"""
from __future__ import annotations

import gc
import inspect
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import torch

from quantization import materialize_quantized_model_dir, validate_quantized_model_dir


DEFAULT_HF_MODEL_REPO = "facebook/musicgen-style"
DEFAULT_HF_MODEL_URL = "https://huggingface.co/facebook/musicgen-style"


class GenerationError(RuntimeError):
    """生成流程相关错误。"""


class MusicGenStyleGenerator:
    """AudioCraft MusicGen-Style 的集中适配层。"""

    def __init__(self, model_name: str = DEFAULT_HF_MODEL_REPO, quantized_model_dir: str | None = None):
        if not torch.cuda.is_available():
            raise GenerationError("CUDA 不可用，MusicGen-Style 不支持在本项目中静默切换到 CPU 生成。")

        self.model_name = str(model_name)
        self.quantized_model_dir = str(quantized_model_dir) if quantized_model_dir else None
        self._quantized_temp_dir = None
        if self.quantized_model_dir:
            validate_quantized_model_dir(self.quantized_model_dir)

        try:
            from audiocraft.models import MusicGen
        except ImportError as exc:
            raise GenerationError("无法导入 audiocraft，请先安装: pip install git+https://github.com/facebookresearch/audiocraft.git") from exc

        try:
            load_name = self.model_name
            if self.quantized_model_dir:
                self._quantized_temp_dir, materialized_dir = materialize_quantized_model_dir(self.quantized_model_dir)
                load_name = str(materialized_dir)
            self.model = self._load_model(MusicGen, load_name)
        except Exception as exc:
            raise GenerationError(f"模型加载失败: {self.model_name}\n{exc}") from exc

        self.doctor()

    def _load_model(self, musicgen_cls, model_name: str):
        """按当前 AudioCraft 签名加载模型。"""
        get_pretrained = musicgen_cls.get_pretrained
        signature = inspect.signature(get_pretrained)
        if "device" in signature.parameters:
            return get_pretrained(model_name, device="cuda")
        return get_pretrained(model_name)

    @property
    def sample_rate(self) -> int:
        sample_rate = getattr(self.model, "sample_rate", None)
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise GenerationError("模型没有有效 sample_rate，无法处理音频。")
        return sample_rate

    @property
    def min_style_window_seconds(self) -> float:
        provider = getattr(getattr(self.model, "lm", None), "condition_provider", None)
        conditioners = getattr(provider, "conditioners", None)
        if not hasattr(conditioners, "__getitem__"):
            raise GenerationError("无法读取模型 condition_provider.conditioners，不能确定真实 style window 最短长度。")
        try:
            self_wav = conditioners["self_wav"]
        except (KeyError, TypeError) as exc:
            raise GenerationError("当前模型缺少 self_wav style conditioner，无法执行 MusicGen-Style 音频条件生成。")
        if self_wav is None:
            raise GenerationError("当前模型 self_wav style conditioner 为空，无法执行 MusicGen-Style 音频条件生成。")
        length_subwav = getattr(self_wav, "length_subwav", None)
        sample_rate = getattr(self_wav, "sample_rate", None)
        if not isinstance(length_subwav, int) or length_subwav <= 0:
            raise GenerationError("当前 self_wav conditioner 缺少有效 length_subwav。")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise GenerationError("当前 self_wav conditioner 缺少有效 sample_rate。")
        return length_subwav / sample_rate

    def doctor(self) -> None:
        """检查 AudioCraft API 能力是否满足 MusicGen-Style 条件生成。"""
        if not hasattr(self.model, "set_generation_params"):
            raise GenerationError("当前模型对象缺少 set_generation_params。")
        if not hasattr(self.model, "generate_with_chroma"):
            raise GenerationError("当前模型对象缺少 generate_with_chroma，无法执行音频条件生成。")

        set_params = inspect.signature(self.model.set_generation_params).parameters
        required = ["duration", "cfg_coef", "temperature", "top_k"]
        missing = [name for name in required if name not in set_params]
        if missing:
            raise GenerationError(f"set_generation_params 缺少必需参数: {', '.join(missing)}")
        if not self._second_cfg_param_name(set_params):
            raise GenerationError(
                "set_generation_params 不支持 README 提到的 cfg_coef_2，也未暴露 AudioCraft 常见的 cfg_coef_beta，"
                "无法同时控制 style/text conditioning guidance。"
            )
        _ = self.min_style_window_seconds

    def _second_cfg_param_name(self, params: dict[str, inspect.Parameter]) -> str | None:
        """识别第二 CFG 参数名。"""
        if "cfg_coef_2" in params:
            return "cfg_coef_2"
        if "cfg_coef_beta" in params:
            return "cfg_coef_beta"
        return None

    def _set_generation_params(
        self,
        duration: float,
        cfg_coef: float,
        cfg_coef_2: float,
        temperature: float,
        top_k: int,
    ) -> None:
        """按当前 AudioCraft 签名设置生成参数。"""
        signature = inspect.signature(self.model.set_generation_params)
        params = signature.parameters
        second_cfg = self._second_cfg_param_name(params)
        if not second_cfg:
            raise GenerationError("当前 AudioCraft API 不支持第二 CFG 参数，拒绝继续。")

        kwargs = {
            "duration": float(duration),
            "cfg_coef": float(cfg_coef),
            second_cfg: float(cfg_coef_2),
            "temperature": float(temperature),
            "top_k": int(top_k),
        }
        if "use_sampling" in params:
            kwargs["use_sampling"] = True
        if "top_p" in params:
            kwargs["top_p"] = 0.0
        self.model.set_generation_params(**kwargs)

    def generate_segment(
        self,
        prompt: str,
        style_audio: torch.Tensor,
        style_sr: int,
        duration: float,
        cfg_coef: float,
        cfg_coef_2: float,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """使用短 style window 和可选 prompt 生成一个音频段。"""
        if duration <= 0:
            raise GenerationError("单段生成时长必须大于 0。")
        if style_audio.ndim != 2:
            raise GenerationError(f"style_audio 必须是 [channels, samples]，当前形状: {tuple(style_audio.shape)}")

        self._set_generation_params(duration, cfg_coef, cfg_coef_2, temperature, top_k)
        descriptions = [prompt or ""]
        style_batch = style_audio.detach().cpu().float().unsqueeze(0).contiguous()

        try:
            with torch.no_grad():
                output = self.model.generate_with_chroma(descriptions, style_batch, int(style_sr))
        except torch.cuda.OutOfMemoryError as exc:
            self.clear_cuda_cache()
            raise GenerationError(
                "显存不足，请尝试减小 --segment-duration、--style-window、--max-style-windows，"
                "并关闭其他占用 GPU 的程序。"
            ) from exc
        except Exception as exc:
            raise GenerationError(f"MusicGen-Style 生成失败: {exc}") from exc

        if not isinstance(output, torch.Tensor):
            raise GenerationError(f"模型输出不是 torch.Tensor: {type(output)!r}")
        if output.ndim == 3:
            output = output[0]
        elif output.ndim == 1:
            output = output.unsqueeze(0)
        elif output.ndim != 2:
            raise GenerationError(f"不支持的模型输出形状: {tuple(output.shape)}")

        result = output.detach().cpu().float().contiguous()
        del output, style_batch
        self.clear_cuda_cache()
        return result

    @staticmethod
    def clear_cuda_cache() -> None:
        """释放 Python 引用并清理 CUDA 缓存。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_long_audio(
    generator: MusicGenStyleGenerator,
    style_windows: Sequence[torch.Tensor],
    prompt: str,
    total_duration: float,
    segment_duration: float,
    overlap: float,
    cfg_coef: float,
    cfg_coef_2: float,
    temperature: float,
    top_k: int,
    dry_run: bool = False,
    progress: Callable[[int, int, float, float], None] | None = None,
) -> list[torch.Tensor]:
    """串行分段生成，返回 CPU 音频段列表。"""
    if not style_windows:
        raise GenerationError("style_windows 为空，无法生成。")
    if total_duration <= 0:
        raise GenerationError("--duration 必须大于 0。")
    if segment_duration <= 0:
        raise GenerationError("--segment-duration 必须大于 0。")
    if overlap < 0:
        raise GenerationError("--overlap 必须大于等于 0。")
    if overlap >= segment_duration:
        raise GenerationError("--overlap 必须小于 --segment-duration。")

    if dry_run:
        actual_duration = min(segment_duration, 8.0)
        start = time.time()
        segment = generator.generate_segment(
            prompt=prompt,
            style_audio=style_windows[0],
            style_sr=generator.sample_rate,
            duration=actual_duration,
            cfg_coef=cfg_coef,
            cfg_coef_2=cfg_coef_2,
            temperature=temperature,
            top_k=top_k,
        )
        if progress:
            progress(1, 1, actual_duration, time.time() - start)
        return [segment]

    num_segments = max(1, math.ceil(total_duration / segment_duration))
    segments: list[torch.Tensor] = []
    for idx in range(num_segments):
        remaining = max(0.0, total_duration - idx * segment_duration)
        generate_duration = min(segment_duration + overlap, remaining + overlap)
        generate_duration = max(generate_duration, min(segment_duration, total_duration))
        style_window = style_windows[idx % len(style_windows)]

        start = time.time()
        segment = generator.generate_segment(
            prompt=prompt,
            style_audio=style_window,
            style_sr=generator.sample_rate,
            duration=generate_duration,
            cfg_coef=cfg_coef,
            cfg_coef_2=cfg_coef_2,
            temperature=temperature,
            top_k=top_k,
        )
        segments.append(segment)
        generator.clear_cuda_cache()
        if progress:
            progress(idx + 1, num_segments, generate_duration, time.time() - start)

    return segments


def resolve_default_model(preferred_model: str | Path | None = None) -> str:
    """解析模型名或目录；未指定时使用官方 Hugging Face 原始模型。"""
    if preferred_model is None:
        return DEFAULT_HF_MODEL_REPO
    path = Path(preferred_model)
    if path.exists():
        return str(path)
    return str(preferred_model)
