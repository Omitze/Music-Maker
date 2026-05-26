# MusicGen-Style MP3 风格生成器操作手册

本目录是项目的可运行源码目录，包含 CLI、Web UI、音频处理、生成流程和历史量化实验工具。

## 1. 功能

- 输入 MP3 作为风格参考音频。
- 输入可选英文 prompt。
- 输出新的 MP3 音乐。
- 从参考音频中提取短 style window。
- 支持 Gradio 网页界面。
- 支持命令行生成。
- 支持串行分段生成和 crossfade 拼接。
- 默认使用官方原始 MusicGen-Style 模型。

## 2. 模型与许可证

默认模型为官方原始 MusicGen-Style 1.5B：

```text
https://huggingface.co/facebook/musicgen-style
```

运行默认使用 Hugging Face repo id：

```text
facebook/musicgen-style
```

模型权重许可证为 `CC-BY-NC 4.0`，仅适合研究和非商业用途。

## 3. 安装环境

推荐使用 conda 环境文件：

```bash
conda env create -f environment.yml
conda activate lhy_music
```

如果环境已经存在，直接激活：

```bash
source /home/luoyingfeng/anaconda3/etc/profile.d/conda.sh
conda activate lhy_music
```

手动安装方式：

```bash
conda create -n lhy_music python=3.10 -y
conda activate lhy_music
conda install -c conda-forge "ffmpeg=6.1.1" "av=11.0.0" -y
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

当前已验证环境包括：

```text
Python 3.10
CUDA 12.1
PyTorch 2.1.0
torchaudio 2.1.0
torchvision 0.16.0
ffmpeg 6.1.1
PyAV 11.0.0
Gradio 6.14.0
```

## 4. 模型下载与缓存

未指定 `--model` 时，项目默认使用官方 Hugging Face 原始模型：

```text
facebook/musicgen-style
https://huggingface.co/facebook/musicgen-style
```

首次运行时，AudioCraft 会从 Hugging Face 下载模型权重。

如果你已经手动下载模型，或希望使用指定模型目录，可以通过 `--model` 指向本地模型目录：

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --model /path/to/local/musicgen-style
```

Web UI 也支持同样的模型指定方式：

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/web_app.py \
  --host 127.0.0.1 \
  --port 7860 \
  --model /path/to/local/musicgen-style
```

不建议默认使用本项目的 INT8 量化模型，因为实际生成效果不如官方原始模型。

## 5. 启动网页

启动网页：

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/web_app.py \
  --host 127.0.0.1 \
  --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

网页支持：

- 上传输入 MP3。
- 输入英文 prompt。
- 设置生成总时长。
- 设置单段时长。
- 设置 crossfade。
- 设置 style window。
- 设置 CFG、temperature、top-k、seed 和 MP3 码率。
- 生成后在线播放和下载 MP3。

网页未指定 `--model` 时默认使用官方原始模型：

```text
facebook/musicgen-style
https://huggingface.co/facebook/musicgen-style
```

如果启动时传入 `--model /path/to/local/musicgen-style`，网页会使用该本地模型目录。

网页输出默认保存到：

```text
/mnt/luoyingfeng/lhy/music_maker/outputs/web
```

## 6. CLI 环境检查

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --doctor
```

成功时会看到：

```text
doctor 完成: CUDA、ffmpeg、模型加载和 API 检查通过。
```

## 7. CLI 最小生成测试

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py \
  -i reference.mp3 \
  -o output_dryrun.mp3 \
  --dry-run \
  --duration 4 \
  --segment-duration 4 \
  --overlap 0.5 \
  --style-window auto \
  --max-style-windows 1 \
  --prompt "atmospheric electronic music with soft pulses and echoing textures"
```

## 8. CLI 正式生成

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py \
  -i reference.mp3 \
  -o output_60s.mp3 \
  --duration 60 \
  --segment-duration 6 \
  --overlap 1 \
  --style-window auto \
  --max-style-windows 2 \
  --prompt "atmospheric electronic music with soft pulses and echoing textures"
```

## 9. 关键参数

- `-i, --input`：输入 MP3。
- `-o, --output`：输出 MP3。
- `--model`：模型名或本地模型目录；未指定时默认 `facebook/musicgen-style`，指定时使用传入的本地目录或模型名。
- `--prompt`：英文文本描述，可为空。
- `--duration`：输出总时长，默认 `90` 秒。
- `--segment-duration`：单段净贡献时长，越小越省显存。
- `--overlap`：段间交叉淡化秒数。
- `--style-window`：推荐使用 `auto`。
- `--style-hop`：风格窗口滑动步长。
- `--max-style-windows`：最多提取风格窗口数量。
- `--style-start`：固定风格窗口起点，可选。
- `--cfg-coef`：style conditioning guidance。
- `--cfg-coef-2`：text conditioning guidance。
- `--temperature`：采样温度。
- `--top-k`：top-k 采样。
- `--seed`：随机种子。
- `--bitrate`：MP3 输出码率。
- `--keep-wav`：保留中间 WAV。
- `--doctor`：只检查环境和模型 API。
- `--dry-run`：只生成短片段测试。

## 10. 输入 MP3 长度与 style window

输入 MP3 可以是一两分钟甚至更长。项目不会把完整 MP3 直接送入 style conditioner，而是从参考音频中提取短 style window。

当前模型真实最短 style window 是 `3.00` 秒，所以推荐使用：

```bash
--style-window auto
```

## 11. 关于量化模型

本项目曾生成并验证 INT8 磁盘量化模型，但实际生成效果不如官方原始模型，因此不再作为默认推荐。

如果显存不足，可以尝试使用项目提供的量化相关脚本：

```text
musicgen_style_mp3/quantization.py
```

已有量化模型目录可通过 CLI 的 `--quantized-model-dir` 指定：

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py \
  -i reference.mp3 \
  -o output_quant.mp3 \
  --quantized-model-dir /mnt/luoyingfeng/lhy/music_maker/quant_models/musicgen-style-int8 \
  --duration 60 \
  --segment-duration 6 \
  --overlap 1 \
  --style-window auto \
  --max-style-windows 2 \
  --prompt "atmospheric electronic music with soft pulses and echoing textures"
```

量化模型只适合作为历史实验或磁盘占用对比参考：

```text
quant_models/musicgen-style-int8/
```

该量化方案主要减少硬盘占用，不能降低推理显存，因为当前 AudioCraft 加载接口仍需要标准权重结构，运行时会先把 INT8 packed 权重反量化到临时目录再加载。

## 12. GitHub 仓库建议

代码可以上传 GitHub，但不要通过普通 Git 提交以下目录：

```text
models/
quant_models/musicgen-style-int8/
quant_models/musicgen-style-fp16/
audios/
outputs/
exe_file/
```

普通 Git 会阻止大于 `100 MiB` 的文件。模型权重应放在 Hugging Face Hub 或其他模型托管服务。

## 13. 常见问题

### CUDA 不可用

确认已激活 `lhy_music`，并安装 CUDA 版 PyTorch。

### style window 太短

使用：

```bash
--style-window auto
```

### 显存不足

尝试：

```bash
--segment-duration 4 --max-style-windows 1
```

也可以尝试 `quantization.py` 产生的量化模型，并通过 `--quantized-model-dir` 加载。但当前 INT8 磁盘量化方案主要减少硬盘占用，不能保证降低推理显存，最终效果请以实际生成结果和显存占用为准。

### 生成速度慢

MusicGen-Style 1.5B 模型较大，长音频会按分段串行生成，这是正常现象。

## 14. 限制

- MusicGen-Style 不擅长生成真实人声。
- 英文 prompt 通常比中文 prompt 效果更好。
- crossfade 只能缓解段间突兀，不能保证旋律、和声、结构完全连续。
- 不承诺低显存显卡一定可运行。
- 不保证生成内容具有版权安全性。
