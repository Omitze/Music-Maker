# MusicGen-Style MP3 风格生成器 —— 项目实现 Prompt

请实现一个 Python 命令行项目：输入一首 MP3 作为风格参考音频，输入可选英文文本 prompt，输出一首新的 MP3 音乐。项目必须基于 MusicGen-Style，并通过短风格窗口、分段生成、CPU 拼接、可验证的量化权重加载和及时释放 GPU 中间变量来尽量降低显存占用与模型硬盘占用。

---

## 1. 模型事实与约束

以 `models/README.md` 中的 MusicGen-Style 说明为准：

- 模型名称：MusicGen-Style 1.5B。
- 官方代码入口：AudioCraft。
- 官方加载方式优先使用：

```python
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("style")
```

- 模型是 text-and-audio-to-music model。
- 模型可以同时使用文本条件和音频风格条件。
- style conditioner 输入的是几秒钟音乐片段。
- 用户提供的参考 MP3 可以是一两分钟或更长。
- 参考 MP3 的总长度不等于 style conditioner 的输入长度。
- style conditioner 实际接收的必须是从参考 MP3 中截取出的短音频窗口。
- README 中说明 style conditioner 使用几秒钟音乐片段，推荐范围是 `1.5` 到 `4.5` 秒。
- 实现时必须读取当前模型中 style conditioner 的真实最短输入需求，例如 AudioCraft `self_wav.length_subwav / sample_rate`。
- 最终传入模型的 style window 必须同时满足 README 推荐范围和当前模型实际最短输入需求。
- 模型权重许可证是 `CC-BY-NC 4.0`，项目 README 必须明确说明仅适合研究和非商业用途。
- 模型不擅长生成真实人声。
- 英文文本描述效果优于非英文描述。
- README 建议从 `cfg_coef=3` 和 `cfg_coef_2=5` 开始，其中第一个系数增强 style conditioning，第二个系数增强 text conditioning。

不要把完整 MP3 直接传给 style conditioner。
不要把“输入 MP3 可以较长”误解为“style conditioner 可以接收完整 MP3”。
不要伪造 style embedding。
不要写 mock 生成逻辑。
不要静默退化为纯文本生成。
不要静默切换到 CPU 生成。
不要优先使用 `transformers.MusicgenForConditionalGeneration`。

---

## 2. 项目目标

### 输入

- 一个 MP3 文件，作为风格参考音频。
- 一个可选英文文本 prompt。

### 输出

- 一个 MP3 文件。
- 内容应是新生成音乐，而不是复制输入音频。
- 风格应尽量接近参考音频，包括节奏感、配器倾向、音色氛围、能量密度等。

### 默认目标

- 默认输出时长：`90` 秒。
- 默认面向低显存环境设计。
- 不承诺 4GB / 6GB 显卡一定能跑通，但必须尽量降低峰值显存。

---

## 3. 技术路线

### 唯一主路线

使用 AudioCraft 的 MusicGen 接口实现：

```python
from audiocraft.models import MusicGen
```

加载：

```python
model = MusicGen.get_pretrained(model_name)
```

默认：

```text
model_name = "style"
```

如果用户通过 `--model` 指定本地路径或其他模型名，则使用用户传入值。

### API 验证要求

必须实现一个独立的模型适配层，不要把 AudioCraft API 调用散落在业务逻辑中。

模型适配层必须在初始化时验证：

- 模型对象存在 `set_generation_params`。
- 模型对象存在用于音频条件生成的接口。
- 优先按照官方 README 示例使用 `generate_with_chroma(descriptions, audio, sr)`。
- 如果当前 AudioCraft 版本无法支持 MusicGen-Style 的音频条件生成，必须抛出清晰错误。
- 如果当前 API 不支持 README 中提到的 `cfg_coef_2`，不得猜测等价参数；应在错误信息中提示当前 AudioCraft 版本与 README 不匹配。

不要因为 API 不匹配就改用纯文本 `generate()`。
不要因为 API 不匹配就改用 transformers。

### 量化路线原则

量化分为两个不同目标，必须在实现中明确区分：

- 推理显存优化：运行时以更低精度加载或执行模型。
- 硬盘占用优化：离线生成并保存一个可独立加载的量化权重目录。

仅仅在运行时量化，不会减少原始模型缓存的硬盘占用。

如果要减少硬盘占用，必须实现“离线量化转换 + 保存量化产物 + 直接加载量化产物”的闭环：

1. 首次转换时可以加载原始 MusicGen-Style 权重。
2. 转换完成后必须保存一个独立的量化模型目录。
3. 后续推理必须能通过 `--quantized-model-dir` 直接加载量化目录。
4. 加载量化目录时不得先加载完整原始权重。
5. 量化目录中不得重复保存完整 FP32 / FP16 原始权重。
6. 量化目录必须包含 `manifest.json`，记录源模型、量化方式、量化后 dtype、AudioCraft 版本、PyTorch 版本、创建时间和验证状态。
7. 量化目录保存完成后，必须重新加载该目录并执行 `doctor` 或 `dry-run` 验证。
8. 验证失败时不得把该目录标记为可用。

不要实现“看起来有量化参数但实际仍加载原始权重”的假量化。
不要把运行时量化误写成可以减少硬盘占用。
不要自动删除原始模型缓存；如需清理，只在 README 中说明由用户手动确认后处理。

---

## 4. 核心设计

### 4.1 MP3 输入输出

用户层面只暴露 MP3 输入和 MP3 输出：

```bash
python main.py -i reference.mp3 -o output.mp3
```

内部允许使用临时 WAV / tensor，但最终产物必须是 MP3。

MP3 解码和编码统一使用 `ffmpeg`，避免依赖不稳定的音频后端差异。

要求：

- 启动时检查 `ffmpeg` 是否存在。
- 输入文件必须存在。
- 输入文件扩展名必须是 `.mp3`。
- 输出文件扩展名必须是 `.mp3`。
- 解码失败要直接报错。
- 编码失败要打印 `ffmpeg` 的 stderr。

---

### 4.2 参考 MP3 与风格窗口提取

用户输入 MP3 是风格来源，可以是一两分钟，也可以更长。

但 MusicGen-Style 的 style conditioner 不是接收完整 MP3，而是接收从参考 MP3 中截取出的短音频窗口。

因此必须明确区分：

- 参考 MP3 长度：用户输入文件的总时长。
- style window 长度：实际传给 style conditioner 的短片段时长。

实现必须先解码完整参考 MP3，再从中按规则提取短 style windows。

默认参数：

```text
--style-window auto
--style-hop 4.0
--max-style-windows 4
```

约束：

- `--style-window` 支持 `auto` 或显式秒数。
- `auto` 表示根据当前模型真实最短输入需求自动选择合法窗口长度。
- 显式秒数必须在 `[1.5, 4.5]` 范围内。
- 实现必须从模型对象中读取 style conditioner 的真实最短输入长度。
- 如果当前模型 `length_subwav / sample_rate = 3.0`，则 `--style-window 1.5` 不允许继续生成，必须报错提示至少使用 `3.0` 秒。
- 最终窗口长度必须满足 `max(README 最小值, 模型真实最小值) <= style_window <= README 最大值`。
- 如果模型真实最小值大于 README 最大值，说明当前模型/API 与 README 不一致，必须报错退出。
- `--style-hop` 必须大于 `0`。
- `--max-style-windows` 必须大于等于 `1`。
- 输入音频总时长必须大于等于 `--style-window`。

滑动窗口策略：

- 如果用户没有指定 `--style-start`，从输入音频开头开始按 `style-hop` 提取窗口。
- 最多提取 `max-style-windows` 个窗口。
- 所有 style windows 只保存在 CPU 内存中。
- 每次生成时只把当前需要的 style window 送入模型。
- 如果输入 MP3 是一两分钟，最多仍只提取 `--max-style-windows` 个短窗口，不要把全部音频送入 GPU。

固定窗口策略：

- 如果用户指定 `--style-start`，只从该时间点截取一个 style window。
- 如果 `style_start + style_window` 超出音频总时长，直接报错。

不要对 style window 做随机扩展、随机裁剪或假数据补齐。
不要为了省事只截取输入 MP3 的前 `1.5` 秒。
不要把用户提供的一两分钟 MP3 整体作为 style condition。

---

### 4.3 分段生成

不得一次性生成完整 90 秒。

默认参数：

```text
--segment-duration 6.0
--overlap 1.0
```

含义：

- `segment-duration` 是每段拼接后对最终音频贡献的净时长。
- 每个实际生成片段可以生成 `segment-duration + overlap` 秒，用于后续 crossfade。
- 最终结果必须裁剪到 `--duration` 指定的准确长度。

约束：

- `--duration` 必须大于 `0`。
- `--segment-duration` 必须大于 `0`。
- `--overlap` 必须大于等于 `0`。
- `--overlap` 必须小于 `--segment-duration`。

段数计算：

```text
num_segments = ceil(duration / segment_duration)
```

风格窗口分配：

- 第 `i` 段使用 `style_windows[i % len(style_windows)]`。
- 这样可以从长参考 MP3 的多个位置提取风格，同时每次只占用一个短风格条件的显存。

每段生成完成后：

- 立即将生成结果转到 CPU。
- 删除 GPU 上的中间变量。
- 调用 `torch.cuda.empty_cache()`。
- 不要在 GPU 上保存历史生成片段。

---

### 4.4 段间拼接

必须在 CPU 上完成拼接。

拼接方式：

- 使用线性交叉淡化。
- `a` 的尾部淡出。
- `b` 的头部淡入。
- overlap 区域相加。

注意：

- crossfade 只能减少 click、爆音和边界突兀。
- crossfade 不能保证音乐语义、旋律、和声和段落结构完全连续。
- README 和日志中不得声称可以完美生成一首结构完整的长音乐。

---

### 4.5 后处理

最终音频保存前必须做：

- 裁剪到目标时长。
- 峰值归一化，避免削波。
- 全局短淡入淡出，默认 `0.5` 秒。

不要做过度压缩。
不要做响度伪增强。
不要改变音乐风格判断所依赖的主要动态。

---

### 4.6 模型量化与量化权重缓存

量化是可选高级能力，但如果实现，必须是完整、可验证、可复现的能力。

支持参数：

```text
--quantization none
--quantization int8
--quantization int4
--prepare-quantized
--quantized-model-dir ./models/musicgen-style-int8
```

默认：

```text
--quantization none
```

要求：

- `none` 表示不量化。
- `int8` / `int4` 只有在当前量化 backend 能真实支持 AudioCraft MusicGen-Style 的加载、保存、重载和推理时才允许启用。
- 不得因为某个 backend 不支持就静默改用 FP16 / FP32。
- 不得静默忽略 `--quantization`。
- 不得静默忽略 `--quantized-model-dir`。
- 不得把原始 checkpoint 复制到量化目录冒充量化产物。
- 不得在量化目录里同时保存完整原始权重和量化权重。
- 不得自动删除 HuggingFace / AudioCraft 原始缓存。

离线量化流程：

```bash
python main.py \
  --prepare-quantized \
  --model style \
  --quantization int8 \
  --quantized-model-dir ./models/musicgen-style-int8
```

离线量化必须执行：

1. 加载原始模型。
2. 检查当前量化 backend 是否支持 AudioCraft MusicGen-Style。
3. 对可安全量化的模块进行量化。
4. 保存量化权重、必要配置和 `manifest.json`。
5. 释放原始模型。
6. 从 `--quantized-model-dir` 重新加载量化模型。
7. 执行 `doctor` 验证。
8. 打印原始模型目录大小、量化模型目录大小和压缩比例。

量化推理流程：

```bash
python main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --quantized-model-dir ./models/musicgen-style-int8
```

量化推理必须直接从 `--quantized-model-dir` 加载量化模型。
如果量化模型加载失败，必须报错退出，不得自动回退到原始模型。

如果无法在当前 AudioCraft 版本上可靠保存和重载量化权重，应直接明确报错：

```text
当前 AudioCraft / 量化 backend 不支持可持久化量化权重，无法减少硬盘占用。
```

---

## 5. 命令行接口

实现 `main.py`。

基础用法：

```bash
python main.py -i reference.mp3 -o output.mp3
```

完整参数：

```bash
python main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --prompt "energetic electronic pop with punchy drums and bright synths" \
  --duration 90 \
  --segment-duration 6 \
  --overlap 1 \
  --style-window auto \
  --style-hop 4 \
  --max-style-windows 4 \
  --cfg-coef 3 \
  --cfg-coef-2 5 \
  --temperature 0.9 \
  --top-k 250 \
  --seed 1234 \
  --bitrate 192k \
  --quantization none
```

参数要求：

- `-i, --input`：输入 MP3，必填。
- `-o, --output`：输出 MP3，必填。
- `--model`：模型名或本地模型路径，默认 `"style"`。
- `--prompt`：可选英文文本描述，默认空字符串。
- `--duration`：输出总时长，默认 `90`。
- `--segment-duration`：单段净贡献时长，默认 `6`。
- `--overlap`：段间交叉淡化秒数，默认 `1`。
- `--style-window`：风格窗口长度，默认 `auto`；显式秒数必须满足 README 范围和当前模型真实最短输入需求。
- `--style-hop`：风格窗口滑动步长，默认 `4`。
- `--max-style-windows`：最多提取窗口数，默认 `4`。
- `--style-start`：可选；指定后只使用一个固定风格窗口。
- `--cfg-coef`：style conditioning guidance，默认 `3`。
- `--cfg-coef-2`：text conditioning guidance，默认 `5`。
- `--temperature`：采样温度，默认 `0.9`。
- `--top-k`：top-k 采样，默认 `250`。
- `--seed`：随机种子，可选。
- `--bitrate`：MP3 输出码率，默认 `192k`。
- `--quantization`：量化方式，默认 `none`；可选 `none`、`int8`、`int4`，只有验证支持后才可启用。
- `--prepare-quantized`：离线生成并保存量化模型目录，不生成音频。
- `--quantized-model-dir`：量化模型目录；生成时指定该参数则必须直接加载量化模型。
- `--keep-wav`：保留中间 WAV。
- `--dry-run`：只生成一个短段，用于环境验证。
- `--doctor`：只检查 CUDA、ffmpeg、模型加载和 API 能力，不生成音频。

---

## 6. 低显存要求

显存优化必须围绕“短窗口 + 短分段 + 串行生成”实现。

必须做到：

1. 不一次性生成完整长音频。
2. 不把完整 MP3 输入 style conditioner。
3. 不在 GPU 上保留全部 style windows。
4. 不在 GPU 上保留全部生成片段。
5. 每次只处理一个 style window 和一个生成段。
6. 每段结束后及时把结果移动到 CPU。
7. 每段结束后清理 CUDA 缓存。
8. 拼接、归一化、淡入淡出、MP3 编码都在 CPU 上完成。
9. 如果启用量化推理，必须直接加载量化权重目录，不得先加载原始权重。

不要实现未经验证的 INT4 / INT8 量化。
不要实现只能运行时量化、不能保存和重载的“硬盘优化”功能。
不要承诺一定适配 4GB 显卡。
不要用 CPU 生成作为“兼容方案”。

OOM 时必须提示用户尝试：

```text
显存不足，请尝试：
1. 减小 --segment-duration，例如 4
2. 使用 --style-window auto，或在模型允许范围内减小 --style-window
3. 减小 --max-style-windows，例如 1
4. 关闭其他占用 GPU 的程序
```

---

## 7. 代码结构

创建如下结构：

```text
musicgen_style_mp3/
├── main.py
├── audio_utils.py
├── generation.py
├── quantization.py
├── requirements.txt
└── README.md
```

### `main.py`

负责：

- 命令行解析。
- 参数校验。
- 环境检查。
- 调用生成流程。
- 打印清晰进度和最终输出路径。

### `audio_utils.py`

负责：

- 检查 `ffmpeg`。
- MP3 解码到临时 WAV。
- WAV 读取到 tensor。
- 按模型采样率重采样。
- 提取 style windows。
- 读取模型真实 style conditioner 最短输入长度。
- CPU crossfade。
- CPU normalize。
- CPU fade in/out。
- 保存临时 WAV。
- WAV 转 MP3。
- 清理临时文件。

### `generation.py`

负责：

- 加载 MusicGen-Style。
- 验证 AudioCraft API。
- 设置生成参数。
- 单段 style-conditioned generation。
- 串行分段生成。
- CUDA 显存释放。

### `quantization.py`

负责：

- 检查量化 backend 能力。
- 离线量化模型。
- 保存量化权重目录。
- 写入和读取 `manifest.json`。
- 直接加载量化权重目录。
- 统计原始模型和量化模型的硬盘占用。
- 验证量化模型可重新加载和推理。

---

## 8. 模型适配层要求

实现：

```python
class MusicGenStyleGenerator:
    def __init__(
        self,
        model_name: str = "style",
        quantized_model_dir: str | None = None,
    ):
        ...

    @property
    def sample_rate(self) -> int:
        ...

    def doctor(self) -> None:
        ...

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
        ...
```

实现要求：

- 如果传入 `quantized_model_dir`，必须直接加载量化目录，不得先加载 `model_name` 的原始权重。
- 如果 `quantized_model_dir` 缺少 `manifest.json` 或验证状态无效，必须拒绝加载。
- 初始化后必须提供方法或属性返回模型真实最短 style window 秒数。
- 如果 AudioCraft 模型中存在 `self_wav.length_subwav`，必须用它计算真实最短窗口。
- 不得硬编码假设最短窗口一定是 `1.5` 秒。
- `style_audio` 形状必须符合 AudioCraft 接口要求，通常应为 `[1, channels, samples]`。
- `descriptions` 必须是列表，例如 `[prompt]`。
- 如果 prompt 为空，传入 `[""]`，不要自行编造默认 prompt。
- 每次调用前根据当前段时长设置 `set_generation_params`。
- 必须显式传入 `duration`、`cfg_coef`、`cfg_coef_2`、`temperature`、`top_k`。
- 如果当前 `set_generation_params` 不接受某个必需参数，直接抛出清晰错误。
- 生成结果必须返回 CPU tensor，形状统一为 `[channels, samples]`。

---

## 9. 生成流程

整体流程：

1. 解析命令行参数。
2. 校验输入输出必须是 MP3。
3. 检查 CUDA 可用；不可用直接退出。
4. 检查 `ffmpeg` 可用；不可用直接退出。
5. 设置随机种子。
6. 如果指定 `--prepare-quantized`，执行离线量化、保存量化目录、重载验证，然后退出。
7. 如果指定 `--quantized-model-dir`，直接加载量化模型目录。
8. 如果未指定 `--quantized-model-dir`，加载 MusicGen-Style 原始模型。
9. 执行 AudioCraft API doctor 检查。
10. 使用 `ffmpeg` 解码输入 MP3 为临时 WAV。
11. 读取 WAV 并重采样到 `model.sample_rate`。
12. 根据 README 范围和模型真实最短输入长度解析 `--style-window`。
13. 提取 style windows。
14. 如果是 `--doctor`，在完成模型和 API 检查后退出。
15. 如果是 `--dry-run`，只生成一个短段并输出 dry-run MP3。
16. 根据 `duration` 和 `segment-duration` 计算段数。
17. 串行生成每一段。
18. 每段生成后移到 CPU 并释放 GPU 中间变量。
19. 在 CPU 上 crossfade 拼接。
20. 裁剪到目标时长。
21. 做全局 fade in/out。
22. 做峰值归一化。
23. 保存临时 WAV。
24. 使用 `ffmpeg` 转码为 MP3。
25. 按 `--keep-wav` 决定是否保留 WAV。
26. 打印输出路径、总耗时、峰值显存。

---

## 10. 错误处理

必须处理并给出清晰错误：

- 输入文件不存在。
- 输入路径不是 `.mp3`。
- 输出路径不是 `.mp3`。
- CUDA 不可用。
- `ffmpeg` 不可用。
- MP3 解码失败。
- 模型加载失败。
- AudioCraft API 不满足 MusicGen-Style 音频条件生成要求。
- 量化 backend 不支持当前模型。
- 量化模型目录不存在。
- 量化模型目录缺少 `manifest.json`。
- 量化模型验证失败。
- 指定了 `--quantized-model-dir` 但代码仍需要原始权重才能加载。
- `style-window` 不在 `[1.5, 4.5]`。
- `style-window` 小于当前模型真实最短输入长度。
- 当前模型真实最短输入长度大于 README 最大推荐窗口。
- 参考音频短于 `style-window`。
- `style-start` 越界。
- `overlap >= segment-duration`。
- 生成过程中 CUDA OOM。
- MP3 编码失败。

不要吞异常。
不要只打印 `Exception` 字符串。
错误信息必须包含用户下一步该怎么调整。

---

## 11. README 要求

README 必须包含：

1. 项目用途。
2. 模型来源。
3. 许可证说明：模型权重 `CC-BY-NC 4.0`，仅适合研究和非商业用途。
4. 安装方式。
5. CUDA 和 ffmpeg 要求。
6. `--doctor` 用法。
7. `--dry-run` 用法。
8. 基础生成用法。
9. 低显存推荐参数。
10. 参考 MP3 可以较长，但实际 style condition 只使用短窗口。
11. `--style-window auto` 的含义。
12. 量化权重保存和加载说明。
13. 运行时量化和可持久化量化权重的区别。
14. 英文 prompt 建议。
15. 模型不能生成真实人声的限制。
16. crossfade 的能力边界。
17. 常见 OOM 调参建议。

安装说明中不要写死不可靠的 PyTorch 版本。
应提示用户按自己的 CUDA 版本安装 PyTorch。

`requirements.txt` 至少包含：

```text
git+https://github.com/facebookresearch/audiocraft.git
numpy
soundfile
tqdm
```

如项目实际使用了其他依赖，必须同步写入 `requirements.txt`。
如果实现量化能力，必须把实际使用的量化 backend 写入 `requirements.txt`，并在 README 中说明该 backend 对 AudioCraft MusicGen-Style 的支持边界。

---

## 12. 验收标准

实现完成后，以下命令必须存在并可用：

### 环境检查

```bash
python main.py -i reference.mp3 -o output.mp3 --doctor
```

### 最小生成测试

```bash
python main.py -i reference.mp3 -o output.mp3 --dry-run
```

### 低显存生成

```bash
python main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --duration 60 \
  --segment-duration 4 \
  --overlap 0.8 \
  --style-window auto \
  --max-style-windows 1
```

### 离线量化权重生成

```bash
python main.py \
  --prepare-quantized \
  --model style \
  --quantization int8 \
  --quantized-model-dir ./models/musicgen-style-int8
```

### 加载量化权重生成

```bash
python main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --quantized-model-dir ./models/musicgen-style-int8 \
  --duration 60 \
  --segment-duration 4 \
  --style-window auto \
  --max-style-windows 1
```

### 标准生成

```bash
python main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --prompt "energetic electronic pop with punchy drums and bright synths" \
  --duration 90 \
  --segment-duration 6 \
  --overlap 1 \
  --style-window auto \
  --style-hop 4 \
  --max-style-windows 4
```

---

## 13. 质量要求

- 代码必须模块化。
- 代码必须简洁。
- 代码必须有必要注释。
- 不要写重复逻辑。
- 不要写 mock 数据。
- 不要伪造模型输出。
- 不要静默忽略 prompt。
- 不要静默忽略 style audio。
- 不要设计纯文本生成降级方案。
- 不要设计 CPU 生成降级方案。
- 不要设计 FP16 / FP32 原始模型降级方案来掩盖量化加载失败。
- 不要把运行时量化描述成可减少硬盘占用。
- 不要保存包含完整原始权重副本的“量化模型目录”。
- 不要把完整输入 MP3 传入 style conditioner。
- 不要限制用户输入 MP3 只能是几秒钟；限制的是传给模型的 style window。
- 不要声称输出音乐具有版权安全保证。
- 不要声称可以完美复刻参考音乐。
- 不要声称可以完美保证长音乐结构连贯。
- 所有外部命令必须捕获 stderr。
- 所有临时文件必须可清理。
- 所有用户参数必须校验。
