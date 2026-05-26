# quant_models Notes

This directory stores historical disk-quantization experiments for MusicGen-Style.

## Current Recommendation

The recommended default model is the official original model:

```text
https://huggingface.co/facebook/musicgen-style
```

The project no longer recommends `musicgen-style-int8` as the default model because its actual generation quality is worse than the official original model.

## Historical Experiment: `musicgen-style-int8`

This is a validated INT8 disk-quantized package.

- Original model directory: `/mnt/luoyingfeng/lhy/music_maker/models/models`
- Quantized model directory: `/mnt/luoyingfeng/lhy/music_maker/quant_models/musicgen-style-int8`
- Original weight size: about `2.9G`
- INT8 packed weight size: about `1.4G`
- Compression ratio: about `48.1%`
- Manifest: `musicgen-style-int8/manifest.json`
- Validated through the project `--doctor` command
- Validated through the project `--dry-run` command

Note: this method mainly reduces disk usage. Because the current AudioCraft loading API still expects the standard checkpoint format, the project dequantizes the INT8 packed weights into a temporary directory before loading. Therefore, this does not reduce inference VRAM usage.

## Reproduce the Experiment If Needed

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py   -i reference.mp3   -o output_quant.mp3   --quantized-model-dir /mnt/luoyingfeng/lhy/music_maker/quant_models/musicgen-style-int8   --duration 60   --segment-duration 6   --overlap 1   --style-window auto   --max-style-windows 2   --prompt "atmospheric electronic music with soft pulses and echoing textures"
```

## `musicgen-style-fp16`

This directory is an FP16 disk-compression experiment. Since the main original checkpoint was already FP16, the total size only changed from about `2.9G` to about `2.8G`, so the benefit is small.
