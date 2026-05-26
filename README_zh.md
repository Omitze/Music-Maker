# MusicGen-Style MP3 Generator

基于 AudioCraft MusicGen-Style 的 MP3 风格参考音乐生成项目。项目支持上传或指定一段 MP3 作为风格参考，再结合可选英文文本描述生成新的 MP3 音乐。

## 功能概览

- **MP3 风格参考**：从输入 MP3 中提取短 style window，而不是把整首歌直接送入模型。
- **文本控制**：支持可选英文 prompt。
- **网页界面**：提供 Gradio Web UI，可上传 MP3、设置生成时长和采样参数。
- **命令行界面**：支持 `--doctor`、`--dry-run` 和正式生成。
- **低显存流程**：使用串行分段生成和 crossfade 拼接。
- **默认模型**：默认使用官方原始模型 `facebook/musicgen-style`。

## 项目入口

详细安装、模型下载、网页启动和 CLI 使用说明请看：

```text
musicgen_style_mp3/README.md
```

English README:

```text
README.en.md
```

## 模型地址

官方原始模型 URL：

```text
https://huggingface.co/facebook/musicgen-style
```

首次运行时会通过 Hugging Face 下载模型权重。你也可以手动下载模型，并在 CLI 中通过 `--model` 指向本地模型目录。

## GitHub 仓库说明

本仓库建议只提交代码、环境文件和说明文档，不提交大模型、输入音频或生成结果。

`.gitignore` 应排除：

```text
models/
audios/
outputs/
exe_file/
```

模型权重体积较大，不建议通过普通 Git 提交。需要分发模型时，优先使用 Hugging Face Hub 或其他模型托管服务。

## 许可证与限制

- **模型许可证**：MusicGen-Style 权重遵循 `CC-BY-NC 4.0`，仅适合研究和非商业用途。
- **生成限制**：模型不擅长真实人声；长音频结构连续性不能完全保证。
- **版权提示**：项目不保证生成内容具有版权安全性。
