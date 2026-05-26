#!/usr/bin/env python3
"""MusicGen-Style MP3 风格生成器网页入口。"""
from __future__ import annotations

import argparse
import random
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch

from audio_utils import (
    AudioProcessingError,
    apply_fade,
    check_ffmpeg,
    concatenate_with_crossfade,
    crop_audio,
    decode_mp3_to_wav,
    encode_wav_to_mp3,
    extract_style_windows,
    load_wav_tensor,
    normalize_peak,
    require_mp3_path,
    resolve_style_window_seconds,
    save_wav_tensor,
)
from generation import DEFAULT_HF_MODEL_REPO, DEFAULT_HF_MODEL_URL, GenerationError, MusicGenStyleGenerator, generate_long_audio, resolve_default_model


PROJECT_ROOT = Path(__file__).resolve().parent
MUSIC_MAKER_ROOT = PROJECT_ROOT.parent
WEB_OUTPUT_DIR = MUSIC_MAKER_ROOT / "outputs" / "web"
GENERATION_LOCK = threading.Lock()


class WebAppError(RuntimeError):
    """网页参数或运行流程错误。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MusicGen-Style MP3 网页生成器")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio 监听地址")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 监听端口")
    parser.add_argument("--share", action="store_true", help="是否启用 Gradio share 链接")
    parser.add_argument("--model", default=None, help=f"模型名或本地模型目录；未指定时使用 {DEFAULT_HF_MODEL_REPO} ({DEFAULT_HF_MODEL_URL})")
    return parser.parse_args()


def set_seed(seed: int | None) -> None:
    """设置生成随机种子。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_cuda() -> None:
    """确认 CUDA 可用，避免静默切换到 CPU。"""
    if not torch.cuda.is_available():
        raise WebAppError("CUDA 不可用。请在 lhy_music 环境中使用 CUDA 版 PyTorch 运行网页。")


def parse_optional_float(value: str | float | int | None, label: str) -> float | None:
    """解析可留空的浮点参数。"""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise WebAppError(f"{label} 必须为空或合法数字。") from exc


def parse_optional_int(value: str | float | int | None, label: str) -> int | None:
    """解析可留空的整数参数。"""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WebAppError(f"{label} 必须为空或合法整数。") from exc


def resolve_uploaded_mp3(uploaded_file: Any) -> Path:
    """解析 Gradio 上传文件路径并校验 MP3。"""
    if uploaded_file is None:
        raise WebAppError("请先上传输入 MP3。")
    if isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
    else:
        name = getattr(uploaded_file, "name", None)
        if not name:
            raise WebAppError("无法读取上传文件路径。")
        path = Path(name)
    path = require_mp3_path(path, "输入文件")
    if not path.exists():
        raise WebAppError(f"上传文件不存在: {path}")
    return path


@lru_cache(maxsize=1)
def get_generator(model_name: str) -> MusicGenStyleGenerator:
    """缓存模型，避免每次网页点击都重新加载权重。"""
    check_ffmpeg()
    ensure_cuda()
    return MusicGenStyleGenerator(model_name=model_name)


def validate_generation_params(
    duration: float,
    segment_duration: float,
    overlap: float,
    style_hop: float,
    max_style_windows: int,
    cfg_coef: float,
    cfg_coef_2: float,
    temperature: float,
    top_k: int,
) -> None:
    """校验网页表单中的生成参数。"""
    if duration <= 0:
        raise WebAppError("生成时长必须大于 0。")
    if segment_duration <= 0:
        raise WebAppError("单段时长必须大于 0。")
    if overlap < 0:
        raise WebAppError("overlap 必须大于等于 0。")
    if overlap >= segment_duration:
        raise WebAppError("overlap 必须小于单段时长。")
    if style_hop <= 0:
        raise WebAppError("style hop 必须大于 0。")
    if max_style_windows < 1:
        raise WebAppError("max style windows 必须大于等于 1。")
    if cfg_coef <= 0 or cfg_coef_2 <= 0:
        raise WebAppError("CFG 参数必须大于 0。")
    if temperature <= 0:
        raise WebAppError("temperature 必须大于 0。")
    if top_k <= 0:
        raise WebAppError("top_k 必须大于 0。")


def generate_from_web(
    model_name: str,
    input_mp3: Any,
    prompt: str,
    duration: float,
    segment_duration: float,
    overlap: float,
    style_window: str,
    style_hop: float,
    max_style_windows: int,
    style_start: str,
    cfg_coef: float,
    cfg_coef_2: float,
    temperature: float,
    top_k: int,
    seed: str,
    bitrate: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str, str]:
    """执行一次网页生成任务，返回音频播放路径、下载路径和状态信息。"""
    if not GENERATION_LOCK.acquire(blocking=False):
        raise gr.Error("已有生成任务正在运行，请等待当前任务完成。")

    try:
        progress(0.02, desc="校验参数")
        input_path = resolve_uploaded_mp3(input_mp3)
        parsed_style_start = parse_optional_float(style_start, "style start")
        parsed_seed = parse_optional_int(seed, "seed")
        max_style_windows = int(max_style_windows)
        top_k = int(top_k)
        validate_generation_params(
            duration=duration,
            segment_duration=segment_duration,
            overlap=overlap,
            style_hop=style_hop,
            max_style_windows=max_style_windows,
            cfg_coef=cfg_coef,
            cfg_coef_2=cfg_coef_2,
            temperature=temperature,
            top_k=top_k,
        )

        progress(0.08, desc="加载官方原始模型")
        check_ffmpeg()
        ensure_cuda()
        set_seed(parsed_seed)
        generator = get_generator(model_name)

        WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = WEB_OUTPUT_DIR / f"musicgen_style_{timestamp}_{uuid.uuid4().hex[:8]}.mp3"

        with tempfile.TemporaryDirectory(prefix="musicgen_style_web_") as temp_dir:
            temp_root = Path(temp_dir)
            decoded_wav = temp_root / "reference.wav"
            output_wav = temp_root / "output.wav"

            progress(0.18, desc="解码参考 MP3")
            decode_mp3_to_wav(input_path, decoded_wav, generator.sample_rate)
            reference_audio, sr = load_wav_tensor(decoded_wav, generator.sample_rate)

            progress(0.28, desc="提取 style window")
            resolved_style_window = resolve_style_window_seconds(style_window, generator.min_style_window_seconds)
            style_windows = extract_style_windows(
                reference_audio,
                sample_rate=sr,
                style_window=resolved_style_window,
                style_hop=style_hop,
                max_style_windows=max_style_windows,
                style_start=parsed_style_start,
            )

            def on_segment(segment_index: int, num_segments: int, current_duration: float, elapsed: float) -> None:
                segment_progress = 0.35 + 0.45 * segment_index / max(num_segments, 1)
                progress(
                    segment_progress,
                    desc=f"生成第 {segment_index}/{num_segments} 段，当前段 {current_duration:.2f}s，耗时 {elapsed:.1f}s",
                )

            progress(0.35, desc="开始生成音乐")
            segments = generate_long_audio(
                generator=generator,
                style_windows=style_windows,
                prompt=prompt or "",
                total_duration=duration,
                segment_duration=segment_duration,
                overlap=overlap,
                cfg_coef=cfg_coef,
                cfg_coef_2=cfg_coef_2,
                temperature=temperature,
                top_k=top_k,
                dry_run=False,
                progress=on_segment,
            )

            progress(0.85, desc="拼接、淡入淡出和归一化")
            full = concatenate_with_crossfade(segments, generator.sample_rate, overlap)
            full = crop_audio(full, generator.sample_rate, duration)
            full = apply_fade(full, generator.sample_rate, fade_seconds=0.5)
            full = normalize_peak(full, peak=0.95)
            save_wav_tensor(full, output_wav, generator.sample_rate)

            progress(0.95, desc="编码 MP3")
            encode_wav_to_mp3(output_wav, output_path, bitrate=bitrate)

        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        status = (
            f"生成完成。\n\n"
            f"- 输出文件：`{output_path}`\n"
            f"- 输入参考：`{input_path.name}`\n"
            f"- 实际 style window：`{resolved_style_window:.2f}s`\n"
            f"- style window 数量：`{len(style_windows)}`\n"
            f"- 峰值显存：`{peak_mb:.0f} MB`\n"
            f"- 模型：`{model_name}`"
        )
        progress(1.0, desc="完成")
        return str(output_path), str(output_path), status
    except (WebAppError, AudioProcessingError, GenerationError) as exc:
        raise gr.Error(str(exc)) from exc
    finally:
        GENERATION_LOCK.release()


def run_doctor(model_name: str) -> str:
    """在网页中检查 CUDA、ffmpeg、原始模型和 AudioCraft API。"""
    try:
        check_ffmpeg()
        ensure_cuda()
        generator = get_generator(model_name)
        return (
            "doctor 完成。\n\n"
            f"- CUDA：可用\n"
            f"- ffmpeg：可用\n"
            f"- 模型采样率：`{generator.sample_rate} Hz`\n"
            f"- 模型真实最短 style window：`{generator.min_style_window_seconds:.2f}s`\n"
            f"- 模型：`{model_name}`"
        )
    except (WebAppError, AudioProcessingError, GenerationError) as exc:
        raise gr.Error(str(exc)) from exc


def build_interface(model_name: str) -> gr.Blocks:
    """构建 Gradio Blocks 页面。"""
    with gr.Blocks(title="MusicGen-Style MP3 风格生成器") as demo:
        gr.Markdown(
            f"""
# MusicGen-Style MP3 风格生成器

上传一段 MP3 作为风格参考，输入可选英文 prompt，设置生成时长和采样参数后生成新的 MP3。

当前模型：`{model_name}`

未通过 `--model` 指定模型时，默认使用官方 Hugging Face 原始模型 `{DEFAULT_HF_MODEL_REPO}`。

模型地址：{DEFAULT_HF_MODEL_URL}
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_mp3 = gr.File(label="输入 MP3", file_types=[".mp3"], type="filepath")
                prompt = gr.Textbox(
                    label="文本描述 Prompt",
                    value="atmospheric electronic music with soft pulses and echoing textures",
                    lines=3,
                )
                duration = gr.Slider(label="生成总时长 / 秒", minimum=4, maximum=180, value=30, step=1)
                segment_duration = gr.Slider(label="单段时长 / 秒", minimum=3, maximum=12, value=6, step=0.5)
                overlap = gr.Slider(label="段间 crossfade / 秒", minimum=0, maximum=3, value=1, step=0.1)
                with gr.Accordion("高级参数", open=False):
                    style_window = gr.Dropdown(
                        label="style window / 秒",
                        choices=["auto", "3.0", "3.5", "4.0", "4.5"],
                        value="auto",
                        allow_custom_value=True,
                    )
                    style_hop = gr.Slider(label="style hop / 秒", minimum=1, maximum=12, value=4, step=0.5)
                    max_style_windows = gr.Slider(label="最大 style window 数量", minimum=1, maximum=8, value=2, step=1)
                    style_start = gr.Textbox(label="固定 style 起点 / 秒，可留空", value="")
                    cfg_coef = gr.Slider(label="style CFG", minimum=0.1, maximum=10, value=3, step=0.1)
                    cfg_coef_2 = gr.Slider(label="text CFG", minimum=0.1, maximum=10, value=5, step=0.1)
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2, value=0.9, step=0.05)
                    top_k = gr.Slider(label="top_k", minimum=1, maximum=500, value=250, step=1)
                    seed = gr.Textbox(label="随机种子，可留空", value="")
                    bitrate = gr.Dropdown(label="MP3 码率", choices=["128k", "192k", "256k", "320k"], value="192k")
                with gr.Row():
                    doctor_button = gr.Button("检查环境", variant="secondary")
                    generate_button = gr.Button("生成 MP3", variant="primary")
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="生成结果预览", type="filepath")
                output_file = gr.File(label="下载 MP3")
                status = gr.Markdown(label="状态")

        generate_button.click(
            fn=lambda *values: generate_from_web(model_name, *values),
            inputs=[
                input_mp3,
                prompt,
                duration,
                segment_duration,
                overlap,
                style_window,
                style_hop,
                max_style_windows,
                style_start,
                cfg_coef,
                cfg_coef_2,
                temperature,
                top_k,
                seed,
                bitrate,
            ],
            outputs=[output_audio, output_file, status],
        )
        doctor_button.click(fn=lambda: run_doctor(model_name), inputs=[], outputs=[status])
    return demo


def main() -> int:
    args = parse_args()
    model_name = resolve_default_model(args.model)
    demo = build_interface(model_name)
    demo.queue(max_size=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
