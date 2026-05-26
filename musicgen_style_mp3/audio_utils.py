#!/usr/bin/env python3
"""音频输入输出、风格窗口提取与 CPU 后处理工具。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

README_MIN_STYLE_WINDOW = 1.5
README_MAX_STYLE_WINDOW = 4.5


class AudioProcessingError(RuntimeError):
    """音频处理相关错误。"""


def require_mp3_path(path: str | Path, label: str) -> Path:
    """校验路径必须是 MP3。"""
    mp3_path = Path(path)
    if mp3_path.suffix.lower() != ".mp3":
        raise AudioProcessingError(f"{label} 必须是 .mp3 文件: {mp3_path}")
    return mp3_path


def check_ffmpeg() -> str:
    """返回 ffmpeg 可执行文件路径。"""
    executable = shutil.which("ffmpeg")
    if not executable:
        raise AudioProcessingError("ffmpeg 不可用，请先安装 ffmpeg，例如: conda install -c conda-forge ffmpeg")
    return executable


def _run_ffmpeg(args: Sequence[str]) -> None:
    """执行 ffmpeg 并在失败时保留 stderr。"""
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioProcessingError("ffmpeg 不可用，请确认 ffmpeg 已安装并在 PATH 中。") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "无 stderr 输出"
        raise AudioProcessingError(f"ffmpeg 执行失败:\n{stderr}") from exc


def decode_mp3_to_wav(input_mp3: str | Path, output_wav: str | Path, sample_rate: int) -> Path:
    """使用 ffmpeg 将 MP3 解码为指定采样率的 WAV。"""
    ffmpeg = check_ffmpeg()
    input_path = require_mp3_path(input_mp3, "输入文件")
    if not input_path.exists():
        raise AudioProcessingError(f"输入文件不存在: {input_path}")

    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        str(output_path),
    ])
    return output_path


def encode_wav_to_mp3(input_wav: str | Path, output_mp3: str | Path, bitrate: str = "192k") -> Path:
    """使用 ffmpeg 将 WAV 编码为 MP3。"""
    ffmpeg = check_ffmpeg()
    input_path = Path(input_wav)
    output_path = require_mp3_path(output_mp3, "输出文件")
    if not input_path.exists():
        raise AudioProcessingError(f"待编码 WAV 不存在: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg([
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-b:a",
        bitrate,
        str(output_path),
    ])
    return output_path


def load_wav_tensor(path: str | Path, target_sr: int) -> tuple[torch.Tensor, int]:
    """读取 WAV 为 `[channels, samples]` 的 CPU float32 tensor。"""
    wav_path = Path(path)
    if not wav_path.exists():
        raise AudioProcessingError(f"WAV 文件不存在: {wav_path}")

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(np.ascontiguousarray(audio.T))
    if sr != target_sr:
        tensor = F.resample(tensor, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    return tensor.contiguous().cpu(), sr


def save_wav_tensor(audio: torch.Tensor, path: str | Path, sample_rate: int) -> Path:
    """保存 `[channels, samples]` tensor 为 WAV。"""
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = _ensure_channel_first(audio).detach().cpu().float()
    data = waveform.T.numpy()
    sf.write(str(wav_path), data, sample_rate)
    return wav_path


def _ensure_channel_first(audio: torch.Tensor) -> torch.Tensor:
    """统一音频 tensor 形状为 `[channels, samples]`。"""
    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.ndim == 2:
        return audio
    if audio.ndim == 3 and audio.shape[0] == 1:
        return audio[0]
    raise AudioProcessingError(f"不支持的音频形状: {tuple(audio.shape)}")


def resolve_style_window_seconds(style_window: str | float, model_min_seconds: float) -> float:
    if model_min_seconds <= 0:
        raise AudioProcessingError(f"模型真实最短 style window 无效: {model_min_seconds}")
    min_allowed = max(README_MIN_STYLE_WINDOW, float(model_min_seconds))
    if min_allowed > README_MAX_STYLE_WINDOW:
        raise AudioProcessingError(
            f"当前模型真实最短 style window 为 {model_min_seconds:.2f}s，"
            f"超过 README 推荐最大值 {README_MAX_STYLE_WINDOW:.2f}s。"
        )
    if isinstance(style_window, str) and style_window.lower() == "auto":
        return min_allowed
    try:
        requested = float(style_window)
    except (TypeError, ValueError) as exc:
        raise AudioProcessingError("--style-window 必须是 auto 或秒数。") from exc
    if not README_MIN_STYLE_WINDOW <= requested <= README_MAX_STYLE_WINDOW:
        raise AudioProcessingError(
            f"--style-window 必须在 [{README_MIN_STYLE_WINDOW}, {README_MAX_STYLE_WINDOW}] 秒范围内。"
        )
    if requested + 1e-6 < min_allowed:
        raise AudioProcessingError(
            f"--style-window={requested:.2f}s 小于当前模型真实最短需求 {min_allowed:.2f}s，"
            "请使用 --style-window auto 或增大该参数。"
        )
    return requested


def extract_style_windows(
    audio: torch.Tensor,
    sample_rate: int,
    style_window: float,
    style_hop: float,
    max_style_windows: int,
    style_start: float | None = None,
) -> list[torch.Tensor]:
    """从参考音频中提取 MusicGen-Style 所需的短风格窗口。"""
    if not README_MIN_STYLE_WINDOW <= style_window <= README_MAX_STYLE_WINDOW:
        raise AudioProcessingError(
            f"--style-window 必须在 [{README_MIN_STYLE_WINDOW}, {README_MAX_STYLE_WINDOW}] 秒范围内。"
        )
    if style_hop <= 0:
        raise AudioProcessingError("--style-hop 必须大于 0。")
    if max_style_windows < 1:
        raise AudioProcessingError("--max-style-windows 必须大于等于 1。")

    waveform = _ensure_channel_first(audio).detach().cpu().float()
    total_samples = waveform.shape[-1]
    window_samples = int(round(style_window * sample_rate))
    if total_samples < window_samples:
        total_sec = total_samples / sample_rate
        raise AudioProcessingError(f"参考音频过短: {total_sec:.2f}s，小于 style-window={style_window:.2f}s。")

    starts: list[int] = []
    if style_start is not None:
        if style_start < 0:
            raise AudioProcessingError("--style-start 必须大于等于 0。")
        start = int(round(style_start * sample_rate))
        if start + window_samples > total_samples:
            raise AudioProcessingError("--style-start + --style-window 超出参考音频总时长。")
        starts.append(start)
    else:
        hop_samples = int(round(style_hop * sample_rate))
        current = 0
        while current + window_samples <= total_samples and len(starts) < max_style_windows:
            starts.append(current)
            current += hop_samples

    if not starts:
        raise AudioProcessingError("未能提取任何 style window，请检查参考音频长度和参数。")

    return [waveform[..., start:start + window_samples].clone().contiguous() for start in starts]


def linear_crossfade(a: torch.Tensor, b: torch.Tensor, sample_rate: int, overlap: float) -> torch.Tensor:
    """CPU 线性交叉淡化拼接两个 `[channels, samples]` 音频段。"""
    first = _ensure_channel_first(a).detach().cpu().float()
    second = _ensure_channel_first(b).detach().cpu().float()
    if first.shape[0] != second.shape[0]:
        raise AudioProcessingError(f"声道数不一致，无法 crossfade: {first.shape[0]} vs {second.shape[0]}")

    n = int(round(overlap * sample_rate))
    n = min(n, first.shape[-1], second.shape[-1])
    if n <= 0:
        return torch.cat([first, second], dim=-1)

    fade_out = torch.linspace(1.0, 0.0, n, dtype=first.dtype).unsqueeze(0)
    fade_in = torch.linspace(0.0, 1.0, n, dtype=second.dtype).unsqueeze(0)
    mixed = first[..., -n:] * fade_out + second[..., :n] * fade_in
    return torch.cat([first[..., :-n], mixed, second[..., n:]], dim=-1)


def concatenate_with_crossfade(segments: Sequence[torch.Tensor], sample_rate: int, overlap: float) -> torch.Tensor:
    """将多个音频段在 CPU 上交叉淡化拼接。"""
    if not segments:
        raise AudioProcessingError("没有可拼接的生成片段。")
    full = _ensure_channel_first(segments[0]).detach().cpu().float()
    for segment in segments[1:]:
        full = linear_crossfade(full, segment, sample_rate, overlap)
    return full


def crop_audio(audio: torch.Tensor, sample_rate: int, duration: float) -> torch.Tensor:
    """裁剪到目标时长。"""
    target_samples = int(round(duration * sample_rate))
    return _ensure_channel_first(audio)[..., :target_samples]


def apply_fade(audio: torch.Tensor, sample_rate: int, fade_seconds: float = 0.5) -> torch.Tensor:
    """全局短淡入淡出。"""
    waveform = _ensure_channel_first(audio).detach().cpu().float().clone()
    n = int(round(fade_seconds * sample_rate))
    if n <= 0 or waveform.shape[-1] < 2 * n:
        return waveform
    fade_in = torch.linspace(0.0, 1.0, n, dtype=waveform.dtype).unsqueeze(0)
    fade_out = torch.linspace(1.0, 0.0, n, dtype=waveform.dtype).unsqueeze(0)
    waveform[..., :n] *= fade_in
    waveform[..., -n:] *= fade_out
    return waveform


def normalize_peak(audio: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    """峰值归一化，避免削波。"""
    waveform = _ensure_channel_first(audio).detach().cpu().float().clone()
    current_peak = waveform.abs().max().item()
    if current_peak > 0:
        waveform = waveform / current_peak * peak
    return waveform
