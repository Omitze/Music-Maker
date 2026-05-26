#!/usr/bin/env python3
"""将已验证的 INT8 量化模型上传到 Hugging Face Hub。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


PROJECT_ROOT = Path(__file__).resolve().parent
MUSIC_MAKER_ROOT = PROJECT_ROOT.parent
DEFAULT_QUANTIZED_MODEL_DIR = MUSIC_MAKER_ROOT / "quant_models" / "musicgen-style-int8"
HF_TOKEN = ""
HF_MODEL_REPO_ID = ""


class UploadError(RuntimeError):
    """Hugging Face 上传流程错误。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="上传 MusicGen-Style INT8 量化模型到 Hugging Face Hub")
    parser.add_argument("--model-dir", default=str(DEFAULT_QUANTIZED_MODEL_DIR), help="本地量化模型目录")
    parser.add_argument("--repo-id", default=HF_MODEL_REPO_ID, help="Hugging Face 模型仓库 ID，例如 username/musicgen-style-int8")
    parser.add_argument("--token", default=HF_TOKEN, help="Hugging Face access token，默认留空")
    parser.add_argument("--private", action="store_true", help="创建私有模型仓库")
    parser.add_argument("--revision", default="main", help="上传分支或 revision")
    parser.add_argument("--dry-run", action="store_true", help="只检查待上传文件，不访问 Hugging Face")
    return parser.parse_args()


def directory_size_bytes(path: Path) -> int:
    """统计目录中文件总大小。"""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_model_dir(model_dir: Path) -> None:
    """校验量化模型目录是否完整且已验证。"""
    if not model_dir.exists() or not model_dir.is_dir():
        raise UploadError(f"量化模型目录不存在: {model_dir}")
    required_files = [
        "manifest.json",
        "state_dict.bin.int8.pt",
        "compression_state_dict.bin.int8.pt",
        "README.md",
    ]
    missing = [filename for filename in required_files if not (model_dir / filename).exists()]
    if missing:
        raise UploadError(f"量化模型目录缺少文件: {', '.join(missing)}")

    manifest_path = model_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UploadError(f"manifest.json 不是合法 JSON: {manifest_path}") from exc

    if manifest.get("quantization") != "int8":
        raise UploadError("manifest.json 中的 quantization 不是 int8，拒绝上传。")
    if not manifest.get("validated"):
        raise UploadError("manifest.json 标记为未验证，拒绝上传。")
    if not manifest.get("reduces_disk_usage"):
        raise UploadError("manifest.json 未标记 reduces_disk_usage，拒绝上传。")


def list_files(model_dir: Path) -> list[Path]:
    """列出待上传文件。"""
    return sorted(path for path in model_dir.rglob("*") if path.is_file())


def print_upload_plan(model_dir: Path) -> None:
    """打印上传计划。"""
    files = list_files(model_dir)
    print(f"量化模型目录: {model_dir}")
    print(f"文件总大小: {directory_size_bytes(model_dir) / 1024**3:.2f} GiB")
    print("待上传文件:")
    for path in files:
        print(f"- {path.relative_to(model_dir)} ({path.stat().st_size / 1024**2:.2f} MiB)")


def upload_model_dir(model_dir: Path, repo_id: str, token: str, private: bool, revision: str) -> None:
    """创建 Hugging Face 模型仓库并上传目录。"""
    if not repo_id.strip():
        raise UploadError("缺少 --repo-id。请填写 Hugging Face 模型仓库 ID，例如 username/musicgen-style-int8。")
    if not token.strip():
        raise UploadError("缺少 --token。请填写 Hugging Face access token，脚本不会内置 token。")

    create_repo(repo_id=repo_id, token=token, private=private, repo_type="model", exist_ok=True)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message="Upload validated MusicGen-Style INT8 quantized model",
    )
    print(f"上传完成: https://huggingface.co/{repo_id}")


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    try:
        validate_model_dir(model_dir)
        print_upload_plan(model_dir)
        if args.dry_run:
            print("dry-run 完成，未访问 Hugging Face。")
            return 0
        upload_model_dir(
            model_dir=model_dir,
            repo_id=str(args.repo_id),
            token=str(args.token),
            private=bool(args.private),
            revision=str(args.revision),
        )
        return 0
    except UploadError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
