# MusicGen-Style MP3 Style Generator Manual

This directory contains the runnable source code for the project, including the CLI, Web UI, audio processing utilities, generation workflow, and historical quantization experiment tools.

## 1. Features

- Use an input MP3 as the style reference audio.
- Enter an optional English prompt.
- Generate a new MP3 music file.
- Extract short style windows from the reference audio.
- Support a Gradio Web UI.
- Support command-line generation.
- Support serial segmented generation and crossfade stitching.
- Use the official original MusicGen-Style model by default.

## 2. Model and License

The default model is the official original MusicGen-Style 1.5B model:

```text
https://huggingface.co/facebook/musicgen-style
```

By default, the runtime uses the following Hugging Face repo id:

```text
facebook/musicgen-style
```

The model weights are licensed under `CC-BY-NC 4.0` and are suitable only for research and non-commercial use.

## 3. Environment Setup

Using the conda environment file is recommended:

```bash
conda env create -f environment.yml
conda activate lhy_music
```

If the environment already exists, activate it directly:

```bash
source /home/luoyingfeng/anaconda3/etc/profile.d/conda.sh
conda activate lhy_music
```

Manual installation:

```bash
conda create -n lhy_music python=3.10 -y
conda activate lhy_music
conda install -c conda-forge "ffmpeg=6.1.1" "av=11.0.0" -y
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The currently verified environment includes:

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

## 4. Model Download and Cache

When `--model` is not specified, the project uses the official original Hugging Face model by default:

```text
facebook/musicgen-style
https://huggingface.co/facebook/musicgen-style
```

On the first run, AudioCraft downloads the model weights from Hugging Face.

If you have manually downloaded the model, or if you want to use a specific model directory, point `--model` to the local model directory:

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --model /path/to/local/musicgen-style
```

The Web UI supports the same model selection method:

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/web_app.py \
  --host 127.0.0.1 \
  --port 7860 \
  --model /path/to/local/musicgen-style
```

Using this project's INT8 quantized model as the default is not recommended, because its actual generation quality is worse than the official original model.

## 5. Start the Web UI

Start the Web UI:

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/web_app.py \
  --host 127.0.0.1 \
  --port 7860
```

Open in the browser:

```text
http://127.0.0.1:7860
```

The Web UI supports:

- Uploading an input MP3.
- Entering an English prompt.
- Setting the total generation duration.
- Setting the segment duration.
- Setting crossfade.
- Setting the style window.
- Setting CFG, temperature, top-k, seed, and MP3 bitrate.
- Previewing and downloading the generated MP3 after generation.

When `--model` is not specified, the Web UI uses the official original model by default:

```text
facebook/musicgen-style
https://huggingface.co/facebook/musicgen-style
```

If `--model /path/to/local/musicgen-style` is passed when starting the Web UI, the Web UI uses that local model directory.

Web UI outputs are saved by default to:

```text
/mnt/luoyingfeng/lhy/music_maker/outputs/web
```

## 6. CLI Environment Check

```bash
python /mnt/luoyingfeng/lhy/music_maker/musicgen_style_mp3/main.py \
  -i reference.mp3 \
  -o output.mp3 \
  --doctor
```

On success, you will see:

```text
doctor completed: CUDA, ffmpeg, model loading, and API checks passed.
```

## 7. Minimal CLI Generation Test

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

## 8. Full CLI Generation

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

## 9. Key Parameters

- `-i, --input`: Input MP3.
- `-o, --output`: Output MP3.
- `--model`: Model name or local model directory. If not specified, defaults to `facebook/musicgen-style`; if specified, the provided local directory or model name is used.
- `--prompt`: English text description. Can be empty.
- `--duration`: Total output duration. Default: `90` seconds.
- `--segment-duration`: Net contribution duration of each segment. Smaller values use less VRAM.
- `--overlap`: Crossfade duration between segments.
- `--style-window`: `auto` is recommended.
- `--style-hop`: Sliding hop duration for style windows.
- `--max-style-windows`: Maximum number of style windows to extract.
- `--style-start`: Optional fixed start time for the style window.
- `--cfg-coef`: Style conditioning guidance.
- `--cfg-coef-2`: Text conditioning guidance.
- `--temperature`: Sampling temperature.
- `--top-k`: Top-k sampling.
- `--seed`: Random seed.
- `--bitrate`: MP3 output bitrate.
- `--keep-wav`: Keep the intermediate WAV file.
- `--doctor`: Only check the environment and model API.
- `--dry-run`: Generate only a short segment for testing.

## 10. Input MP3 Length and Style Window

The input MP3 can be one or two minutes long, or even longer. The project does not feed the full MP3 directly into the style conditioner. Instead, it extracts short style windows from the reference audio.

The actual minimum style window for the current model is `3.00` seconds, so the recommended setting is:

```bash
--style-window auto
```

## 11. About Quantized Models

This project previously generated and verified an INT8 disk-quantized model, but its actual generation quality is worse than the official original model, so it is no longer recommended as the default.

If you run out of VRAM, you can try the quantization-related script provided by this project:

```text
musicgen_style_mp3/quantization.py
```

An existing quantized model directory can be specified through the CLI with `--quantized-model-dir`:

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

The quantized model is only suitable as a historical experiment or as a disk-usage comparison reference:

```text
quant_models/musicgen-style-int8/
```

This quantization approach mainly reduces disk usage and does not reduce inference VRAM usage, because the current AudioCraft loading interface still requires the standard weight structure. At runtime, the INT8 packed weights are first dequantized into a temporary directory and then loaded.

## 12. GitHub Repository Recommendations

The code can be uploaded to GitHub, but do not commit the following directories through regular Git:

```text
models/
quant_models/musicgen-style-int8/
quant_models/musicgen-style-fp16/
audios/
outputs/
exe_file/
```

Regular Git blocks files larger than `100 MiB`. Model weights should be hosted on Hugging Face Hub or another model hosting service.

## 13. FAQ

### CUDA Is Not Available

Make sure `lhy_music` is activated and that the CUDA version of PyTorch is installed.

### Style Window Is Too Short

Use:

```bash
--style-window auto
```

### Out of VRAM

Try:

```bash
--segment-duration 4 --max-style-windows 1
```

You can also try a quantized model produced by `quantization.py` and load it with `--quantized-model-dir`. However, the current INT8 disk quantization approach mainly reduces disk usage and cannot guarantee lower inference VRAM usage. The final result should be judged by the actual generation quality and VRAM usage.

### Generation Is Slow

MusicGen-Style 1.5B is a large model. Long audio is generated serially in segments, which is expected.

## 14. Limitations

- MusicGen-Style is not good at generating realistic vocals.
- English prompts usually work better than Chinese prompts.
- Crossfade can only reduce abrupt transitions between segments; it cannot guarantee full continuity of melody, harmony, or structure.
- There is no guarantee that low-VRAM GPUs can run this project.
- There is no guarantee that generated content is copyright-safe.
