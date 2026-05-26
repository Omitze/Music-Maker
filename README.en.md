# MusicGen-Style MP3 Generator

A MusicGen-Style MP3 generation project based on AudioCraft. It uses an MP3 reference track plus an optional English text prompt to generate a new MP3 music file.

## Features

- **MP3 style reference**: extracts short style windows from the input MP3 instead of feeding the whole track into the style conditioner.
- **Text control**: supports an optional English prompt.
- **Web UI**: provides a Gradio interface for MP3 upload, duration settings, and sampling parameters.
- **CLI**: supports `--doctor`, `--dry-run`, and full generation.
- **Low-VRAM workflow**: uses serial segmented generation and crossfade stitching.
- **Default model**: uses the official original `facebook/musicgen-style` model by default.

## Project Entry Point

For installation, model download, Web UI startup, and CLI usage, see:

```text
musicgen_style_mp3/README.md
```

Chinese README:

```text
README.zh-CN.md
```

## Model URL

Official original model URL:

```text
https://huggingface.co/facebook/musicgen-style
```

The model weights are downloaded from Hugging Face on first use. You can also download the model manually and pass the local model directory with `--model` in the CLI.

## GitHub Repository Notes

This repository should contain only code, environment files, and documentation. Do not commit large model weights, input audio files, or generated outputs.

`.gitignore` should exclude:

```text
models/
audios/
outputs/
exe_file/
```

Model weights are large and should not be committed through regular Git. Use Hugging Face Hub or another model hosting service when distributing model weights.

## License and Limitations

- **Model license**: MusicGen-Style weights use `CC-BY-NC 4.0` and are intended for research and non-commercial use.
- **Generation limitations**: the model is not good at realistic vocals, and long-term musical structure is not guaranteed.
- **Copyright notice**: this project does not guarantee copyright safety for generated content.
