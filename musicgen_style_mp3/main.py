#!/usr/bin/env python3
"""MusicGen-Style MP3 风格生成器命令行入口。"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

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
from quantization import QuantizationError, prepare_quantized_model

PROJECT_ROOT = Path(__file__).resolve().parent


class CliError(RuntimeError):
    """命令行参数或执行流程错误。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MusicGen-Style MP3 风格生成器")
    parser.add_argument("-i", "--input", help="输入 MP3，作为风格参考音频")
    parser.add_argument("-o", "--output", help="输出 MP3")
    parser.add_argument("--model", default=None, help=f"模型名或本地模型目录，默认使用 {DEFAULT_HF_MODEL_REPO} ({DEFAULT_HF_MODEL_URL})")
    parser.add_argument("--prompt", default="", help="可选英文文本描述")
    parser.add_argument("--duration", type=float, default=90.0, help="输出总时长，秒")
    parser.add_argument("--segment-duration", type=float, default=6.0, help="单段净贡献时长，秒")
    parser.add_argument("--overlap", type=float, default=1.0, help="段间交叉淡化秒数")
    parser.add_argument("--style-window", default="auto", help="风格窗口长度，支持 auto 或 [1.5, 4.5] 秒数")
    parser.add_argument("--style-hop", type=float, default=4.0, help="风格窗口滑动步长，秒")
    parser.add_argument("--max-style-windows", type=int, default=4, help="最多提取风格窗口数量")
    parser.add_argument("--style-start", type=float, default=None, help="固定风格窗口起点，秒")
    parser.add_argument("--cfg-coef", type=float, default=3.0, help="style conditioning guidance")
    parser.add_argument("--cfg-coef-2", type=float, default=5.0, help="text conditioning guidance")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--top-k", type=int, default=250, help="top-k 采样")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--bitrate", default="192k", help="MP3 输出码率")
    parser.add_argument("--quantization", choices=["none", "int8", "int4"], default="none", help="量化方式")
    parser.add_argument("--prepare-quantized", action="store_true", help="离线生成并保存量化模型目录，不生成音频")
    parser.add_argument("--quantized-model-dir", default=None, help="量化模型目录；指定后必须直接加载该目录")
    parser.add_argument("--keep-wav", action="store_true", help="保留中间 WAV")
    parser.add_argument("--dry-run", action="store_true", help="只生成一个短段用于环境验证")
    parser.add_argument("--doctor", action="store_true", help="只检查 CUDA、ffmpeg、模型加载和 API 能力")
    return parser.parse_args()


def set_seed(seed: int | None) -> None:
    """设置随机种子。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_common_args(args: argparse.Namespace) -> None:
    """校验通用参数。"""
    if args.duration <= 0:
        raise CliError("--duration 必须大于 0。")
    if args.segment_duration <= 0:
        raise CliError("--segment-duration 必须大于 0。")
    if args.overlap < 0:
        raise CliError("--overlap 必须大于等于 0。")
    if args.overlap >= args.segment_duration:
        raise CliError("--overlap 必须小于 --segment-duration。")
    if isinstance(args.style_window, str) and args.style_window.lower() != "auto":
        try:
            float(args.style_window)
        except ValueError as exc:
            raise CliError("--style-window 必须是 auto 或秒数。") from exc
    if args.style_hop <= 0:
        raise CliError("--style-hop 必须大于 0。")
    if args.max_style_windows < 1:
        raise CliError("--max-style-windows 必须大于等于 1。")
    if args.top_k <= 0:
        raise CliError("--top-k 必须大于 0。")
    if args.temperature <= 0:
        raise CliError("--temperature 必须大于 0。")


def validate_audio_args(args: argparse.Namespace) -> tuple[Path, Path]:
    """校验普通生成需要的输入输出参数。"""
    if not args.input:
        raise CliError("缺少 -i/--input。")
    if not args.output:
        raise CliError("缺少 -o/--output。")
    input_path = require_mp3_path(args.input, "输入文件")
    output_path = require_mp3_path(args.output, "输出文件")
    if not input_path.exists():
        raise CliError(f"输入文件不存在: {input_path}")
    return input_path, output_path


def ensure_cuda() -> None:
    """确认 CUDA 可用。"""
    if not torch.cuda.is_available():
        raise CliError("CUDA 不可用。请安装 CUDA 版 PyTorch，本项目不静默切换到 CPU 生成。")


def print_header(args: argparse.Namespace, model_name: str) -> None:
    """打印运行配置。"""
    print("=" * 60)
    print("MusicGen-Style MP3 风格生成器")
    print("=" * 60)
    print(f"模型: {model_name}")
    if args.input:
        print(f"输入: {args.input}")
    if args.output:
        print(f"输出: {args.output}")
    print(f"时长: {args.duration}s")
    print(f"分段: {args.segment_duration}s + overlap {args.overlap}s")
    print(f"风格窗口: {args.style_window}, hop {args.style_hop}s, max {args.max_style_windows}")
    print(f"Prompt: {args.prompt or '<empty>'}")
    print("=" * 60)


def run_prepare_quantized(args: argparse.Namespace, model_name: str) -> None:
    """执行离线量化准备。"""
    if not args.quantized_model_dir:
        raise CliError("--prepare-quantized 必须指定 --quantized-model-dir。")
    prepare_quantized_model(
        model_name=model_name,
        quantization=args.quantization,
        quantized_model_dir=args.quantized_model_dir,
    )


def build_generator(args: argparse.Namespace, model_name: str) -> MusicGenStyleGenerator:
    """创建模型适配器。"""
    return MusicGenStyleGenerator(
        model_name=model_name,
        quantized_model_dir=args.quantized_model_dir,
    )


def progress(segment_index: int, num_segments: int, duration: float, elapsed: float) -> None:
    """打印生成进度。"""
    print(f"生成进度: {segment_index}/{num_segments}, 当前段 {duration:.2f}s, 耗时 {elapsed:.1f}s")


def run_generation(args: argparse.Namespace, input_path: Path, output_path: Path, model_name: str) -> None:
    """执行 doctor、dry-run 或正式生成。"""
    start_time = time.time()
    check_ffmpeg()
    ensure_cuda()
    set_seed(args.seed)

    print_header(args, model_name)
    print("[1/5] 加载模型并检查 AudioCraft API...")
    generator = build_generator(args, model_name)
    print(f"模型采样率: {generator.sample_rate} Hz")
    print(f"模型真实最短 style window: {generator.min_style_window_seconds:.2f}s")

    if args.doctor:
        print("doctor 完成: CUDA、ffmpeg、模型加载和 API 检查通过。")
        return

    with tempfile.TemporaryDirectory(prefix="musicgen_style_") as temp_dir:
        temp_root = Path(temp_dir)
        decoded_wav = temp_root / "reference.wav"
        output_wav = output_path.with_suffix(".wav") if args.keep_wav else temp_root / "output.wav"

        print("[2/5] 解码并读取参考 MP3...")
        decode_mp3_to_wav(input_path, decoded_wav, generator.sample_rate)
        reference_audio, sr = load_wav_tensor(decoded_wav, generator.sample_rate)
        print(f"参考音频: {reference_audio.shape[-1] / sr:.2f}s, channels={reference_audio.shape[0]}")

        print("[3/5] 提取短风格窗口...")
        style_window = resolve_style_window_seconds(args.style_window, generator.min_style_window_seconds)
        print(f"实际使用 style window: {style_window:.2f}s")
        style_windows = extract_style_windows(
            reference_audio,
            sample_rate=sr,
            style_window=style_window,
            style_hop=args.style_hop,
            max_style_windows=args.max_style_windows,
            style_start=args.style_start,
        )
        print(f"风格窗口数量: {len(style_windows)}")

        print("[4/5] 串行分段生成...")
        segments = generate_long_audio(
            generator=generator,
            style_windows=style_windows,
            prompt=args.prompt,
            total_duration=args.duration,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            cfg_coef=args.cfg_coef,
            cfg_coef_2=args.cfg_coef_2,
            temperature=args.temperature,
            top_k=args.top_k,
            dry_run=args.dry_run,
            progress=progress,
        )

        print("[5/5] CPU 拼接、后处理并编码 MP3...")
        full = concatenate_with_crossfade(segments, generator.sample_rate, args.overlap)
        target_duration = min(args.segment_duration, args.duration) if args.dry_run else args.duration
        full = crop_audio(full, generator.sample_rate, target_duration)
        full = apply_fade(full, generator.sample_rate, fade_seconds=0.5)
        full = normalize_peak(full, peak=0.95)
        save_wav_tensor(full, output_wav, generator.sample_rate)
        encode_wav_to_mp3(output_wav, output_path, bitrate=args.bitrate)

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值显存: {peak_mb:.0f} MB")
    print(f"完成: {output_path}")
    print(f"总耗时: {time.time() - start_time:.1f}s")


def main() -> int:
    args = parse_args()
    model_name = args.model or resolve_default_model()

    try:
        validate_common_args(args)
        if args.prepare_quantized:
            ensure_cuda()
            set_seed(args.seed)
            run_prepare_quantized(args, model_name)
            return 0
        input_path, output_path = validate_audio_args(args)
        run_generation(args, input_path, output_path, model_name)
        return 0
    except (CliError, AudioProcessingError, GenerationError, QuantizationError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
