"""Push the assembled Swift MLX bundle to a HuggingFace model repo.

The bundle dir is whatever `build-bundle` produced (default
`swift/Sources/TinyAudio/Resources/Model`). We generate a README from the
bundle's `config.json` plus the source projector repo id, write it into the
bundle dir, then ensure the target HF repo exists and upload the folder.
"""

from __future__ import annotations

import json
from pathlib import Path


def _render_readme(bundle_dir: Path, projector_repo: str) -> str:
    """Render a model-card README from `bundle_dir/config.json` + `projector_repo`.

    Pure function — reads config.json from disk, returns the README string.
    All quant/dim/token values come from the bundle's own config so the README
    stays consistent with what's actually in the artifact.
    """
    cfg = json.loads((bundle_dir / "config.json").read_text())
    enc = cfg["encoder"]
    q = enc["quantization"]
    return (
        "# tiny-audio Swift bundle\n"
        "\n"
        "MLX bundle for [tiny-audio-swift](https://github.com/alexkroman/tiny-audio-swift), "
        f"built from projector [`{projector_repo}`](https://huggingface.co/{projector_repo}).\n"
        "\n"
        "## Contents\n"
        f"- `encoder.safetensors` — GLM-ASR encoder, quantized {q['bits']}-bit "
        f"(group {q['group_size']})\n"
        "- `projector.safetensors` — fp16 projector\n"
        "- `decoder.safetensors` — MLX-LM decoder\n"
        "- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `decoder_config.json`\n"
        "- `manifest.json` — sha256 + sizes for all files\n"
        "\n"
        "## Model config\n"
        f"- Encoder dim: {enc['encoder_dim']}, layers: {enc['num_layers']}, "
        f"heads: {enc['num_heads']}\n"
        f"- Audio token: `{cfg['audio_token']}` (id {cfg['audio_token_id']})\n"
        f"- Hop length: {cfg['hop_length']}\n"
        f"- Format version: {cfg['mlx_format_version']}\n"
        "\n"
        "## Usage\n"
        "Loaded automatically by tiny-audio-swift's `Transcriber`.\n"
    )


def push_bundle(
    *,
    bundle_dir: Path,
    repo: str,
    projector_repo: str,
    private: bool,
    commit_message: str,
) -> None:
    """Validate the bundle, render+write README, ensure the repo exists, upload.

    Auth comes from `huggingface_hub`'s default token resolution
    (`HF_TOKEN` env var or `huggingface-cli login` credentials).
    """
    from huggingface_hub import HfApi

    if not bundle_dir.is_dir():
        raise SystemExit(f"Bundle directory not found: {bundle_dir}")
    if not (bundle_dir / "manifest.json").exists():
        raise SystemExit(f"{bundle_dir} has no manifest.json — run `ta mlx build-bundle` first.")

    readme = _render_readme(bundle_dir, projector_repo)
    (bundle_dir / "README.md").write_text(readme)

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Pushed to https://huggingface.co/{repo}")
