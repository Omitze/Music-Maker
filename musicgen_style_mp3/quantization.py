#!/usr/bin/env python3
"""MusicGen-Style 可持久化磁盘量化权重。"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SUPPORTED_BACKEND = "musicgen_style_int8_disk_pack_v1"
PACKED_SUFFIX = ".int8.pt"
FLOAT_DTYPES = {
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
}
DTYPE_BY_NAME = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
}


class QuantizationError(RuntimeError):
    """量化流程相关错误。"""


@dataclass(frozen=True)
class QuantizedManifest:
    source_model: str
    quantization: str
    backend: str
    validated: bool
    format_version: int


def directory_size_bytes(path: str | Path) -> int:
    """统计目录硬盘占用。"""
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _quantize_tensor(tensor: torch.Tensor) -> Any:
    if not tensor.is_floating_point() or tensor.dtype not in FLOAT_DTYPES:
        return tensor
    cpu_tensor = tensor.detach().cpu().contiguous()
    max_abs = cpu_tensor.abs().max().item()
    scale = max(max_abs / 127.0, 1e-12)
    quantized = torch.clamp(torch.round(cpu_tensor.float() / scale), -127, 127).to(torch.int8)
    return {
        "__musicgen_int8_tensor__": True,
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "scale": float(scale),
        "data": quantized.contiguous(),
    }


def _dequantize_tensor(record: dict[str, Any]) -> torch.Tensor:
    dtype_name = str(record["dtype"])
    dtype = DTYPE_BY_NAME.get(dtype_name)
    if dtype is None:
        raise QuantizationError(f"不支持的反量化 dtype: {dtype_name}")
    data = record["data"]
    if not torch.is_tensor(data) or data.dtype != torch.int8:
        raise QuantizationError("量化 tensor 记录缺少 int8 data。")
    tensor = data.float() * float(record["scale"])
    return tensor.reshape(record["shape"]).to(dtype).contiguous()


def _walk_quantize(obj: Any, stats: dict[str, int]) -> Any:
    if torch.is_tensor(obj):
        stats[str(obj.dtype)] = stats.get(str(obj.dtype), 0) + obj.numel()
        return _quantize_tensor(obj)
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            obj[key] = _walk_quantize(obj[key], stats)
        return obj
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            obj[idx] = _walk_quantize(item, stats)
        return obj
    if isinstance(obj, tuple):
        return tuple(_walk_quantize(item, stats) for item in obj)
    return obj


def _walk_dequantize(obj: Any, stats: dict[str, int]) -> Any:
    if isinstance(obj, dict) and obj.get("__musicgen_int8_tensor__"):
        tensor = _dequantize_tensor(obj)
        stats[str(tensor.dtype)] = stats.get(str(tensor.dtype), 0) + tensor.numel()
        return tensor
    if torch.is_tensor(obj):
        stats[str(obj.dtype)] = stats.get(str(obj.dtype), 0) + obj.numel()
        return obj
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            obj[key] = _walk_dequantize(obj[key], stats)
        return obj
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            obj[idx] = _walk_dequantize(item, stats)
        return obj
    if isinstance(obj, tuple):
        return tuple(_walk_dequantize(item, stats) for item in obj)
    return obj


def load_manifest(quantized_model_dir: str | Path) -> QuantizedManifest:
    """读取并校验量化模型 manifest。"""
    model_dir = Path(quantized_model_dir)
    if not model_dir.exists():
        raise QuantizationError(f"量化模型目录不存在: {model_dir}")
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise QuantizationError(f"量化模型目录缺少 manifest.json: {manifest_path}")

    try:
        data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QuantizationError(f"manifest.json 不是合法 JSON: {manifest_path}") from exc

    required = ["source_model", "quantization", "backend", "validated", "format_version"]
    missing = [key for key in required if key not in data]
    if missing:
        raise QuantizationError(f"manifest.json 缺少字段: {', '.join(missing)}")
    if data["backend"] != SUPPORTED_BACKEND:
        raise QuantizationError(f"不支持的量化 backend: {data['backend']}")

    return QuantizedManifest(
        source_model=str(data["source_model"]),
        quantization=str(data["quantization"]),
        backend=str(data["backend"]),
        validated=bool(data["validated"]),
        format_version=int(data["format_version"]),
    )


def _write_manifest(model_dir: Path, manifest: dict[str, Any]) -> None:
    (model_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _pack_checkpoint(source_file: Path, target_file: Path) -> dict[str, Any]:
    obj = torch.load(source_file, map_location="cpu")
    source_stats: dict[str, int] = {}
    packed = _walk_quantize(obj, source_stats)
    torch.save(packed, target_file)
    return {
        "source_bytes": _file_size(source_file),
        "target_bytes": _file_size(target_file),
        "source_tensor_numel_by_dtype": source_stats,
    }


def materialize_quantized_model_dir(
    quantized_model_dir: str | Path,
    require_validated: bool = True,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """把 INT8 磁盘量化目录反量化到临时 AudioCraft 模型目录。"""
    model_dir = Path(quantized_model_dir)
    manifest = load_manifest(model_dir)
    if require_validated and not manifest.validated:
        raise QuantizationError("量化模型尚未通过验证，拒绝加载。")
    if manifest.quantization != "int8":
        raise QuantizationError(f"当前只支持 int8 量化目录，收到: {manifest.quantization}")

    temp_dir = tempfile.TemporaryDirectory(prefix="musicgen_style_int8_dequant_")
    target_dir = Path(temp_dir.name)
    for name in ["README.md", ".gitattributes"]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)
    for filename in ["state_dict.bin", "compression_state_dict.bin"]:
        packed_file = model_dir / f"{filename}{PACKED_SUFFIX}"
        if not packed_file.exists():
            temp_dir.cleanup()
            raise QuantizationError(f"量化目录缺少文件: {packed_file}")
        obj = torch.load(packed_file, map_location="cpu")
        stats: dict[str, int] = {}
        dequantized = _walk_dequantize(obj, stats)
        torch.save(dequantized, target_dir / filename)
        del obj, dequantized
    return temp_dir, target_dir


def _validate_materialized_model(model_dir: Path) -> None:
    try:
        from audiocraft.models import MusicGen
    except ImportError as exc:
        raise QuantizationError("无法导入 audiocraft，不能验证量化模型。") from exc
    try:
        model = MusicGen.get_pretrained(str(model_dir), device="cuda" if torch.cuda.is_available() else None)
    except Exception as exc:
        raise QuantizationError(f"量化模型反量化后无法加载: {exc}") from exc
    if not hasattr(model, "set_generation_params") or not hasattr(model, "generate_with_chroma"):
        raise QuantizationError("量化模型反量化后缺少 MusicGen-Style 必需 API。")


def prepare_quantized_model(model_name: str, quantization: str, quantized_model_dir: str | Path) -> None:
    """离线生成可独立加载的 INT8 磁盘量化模型目录。"""
    if quantization == "none":
        raise QuantizationError("--prepare-quantized 需要 --quantization int8。")
    if quantization != "int8":
        raise QuantizationError("当前只实现可验证的 int8 磁盘量化；int4 暂不启用，避免生成不可加载产物。")

    source_dir = Path(model_name)
    if not source_dir.is_dir():
        raise QuantizationError("当前量化实现只接受本地模型目录，请通过 --model 指向 models/models。")
    required_files = ["state_dict.bin", "compression_state_dict.bin"]
    missing = [name for name in required_files if not (source_dir / name).exists()]
    if missing:
        raise QuantizationError(f"本地模型目录缺少权重文件: {', '.join(missing)}")

    target_dir = Path(quantized_model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale in target_dir.glob("*.bin"):
        raise QuantizationError(f"量化目录中存在未量化 .bin 文件，拒绝继续: {stale}")
    for name in ["README.md", ".gitattributes"]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_model": str(source_dir),
        "quantization": "int8",
        "backend": SUPPORTED_BACKEND,
        "validated": False,
        "reduces_disk_usage": True,
        "reduces_inference_vram": False,
        "requires_dequantization_for_audiocraft": True,
        "files": {},
    }

    for filename in required_files:
        source_file = source_dir / filename
        target_file = target_dir / f"{filename}{PACKED_SUFFIX}"
        manifest["files"][filename] = _pack_checkpoint(source_file, target_file)

    source_total = sum(_file_size(source_dir / name) for name in required_files)
    target_total = sum(_file_size(target_dir / f"{name}{PACKED_SUFFIX}") for name in required_files)
    manifest["source_weight_bytes"] = source_total
    manifest["target_weight_bytes"] = target_total
    manifest["compression_ratio"] = target_total / source_total if source_total else None
    _write_manifest(target_dir, manifest)

    temp_dir, materialized_dir = materialize_quantized_model_dir(target_dir, require_validated=False)
    try:
        _validate_materialized_model(materialized_dir)
    finally:
        temp_dir.cleanup()

    manifest["validated"] = True
    manifest["validated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(target_dir, manifest)


def validate_quantized_model_dir(quantized_model_dir: str | Path) -> QuantizedManifest:
    """校验量化目录。"""
    manifest = load_manifest(quantized_model_dir)
    if not manifest.validated:
        raise QuantizationError("量化模型 manifest 标记为未验证，拒绝加载。")
    return manifest
