# quant_models 说明

本目录保存 MusicGen-Style 的历史磁盘量化实验产物。

## 当前推荐

默认推荐使用官方原始模型：

```text
https://huggingface.co/facebook/musicgen-style
```

项目当前不再作为默认推荐使用 `musicgen-style-int8`，因为实际生成效果不如官方原始模型。

## 历史实验：`musicgen-style-int8`

这是已验证的 INT8 磁盘量化包。

- 原始模型目录：`/mnt/luoyingfeng/lhy/music_maker/models/models`
- 量化模型目录：`/mnt/luoyingfeng/lhy/music_maker/quant_models/musicgen-style-int8`
- 原始权重大小：约 `2.9G`
- INT8 packed 权重大小：约 `1.4G`
- 压缩比例：约 `48.1%`
- manifest：`musicgen-style-int8/manifest.json`
- 已通过项目入口 `--doctor` 验证
- 已通过项目入口 `--dry-run` 验证

注意：该方案主要减少硬盘占用。由于 AudioCraft 当前加载接口仍需要标准权重结构，运行时会先把 INT8 packed 权重反量化到临时目录再加载，所以它不降低推理显存。

## 如需复现实验

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py   -i reference.mp3   -o output_quant.mp3   --quantized-model-dir /mnt/luoyingfeng/lhy/music_maker/quant_models/musicgen-style-int8   --duration 60   --segment-duration 6   --overlap 1   --style-window auto   --max-style-windows 2   --prompt "atmospheric electronic music with soft pulses and echoing textures"
```

## `musicgen-style-fp16`

该目录是 FP16 磁盘压缩实验产物，但原始主权重已经是 FP16，因此整体大小只从约 `2.9G` 降到约 `2.8G`，收益很小。
