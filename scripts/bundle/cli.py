"""CLI for MLX bundle build utilities."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(help="MLX bundle build utilities", no_args_is_help=True)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DECODER_CACHE = _PROJECT_ROOT / ".cache" / "decoder-mlx"
_DEFAULT_BUNDLE_DIR = _PROJECT_ROOT / "swift/Sources/TinyAudio/Resources/Model"
_STOCK_DECODER_REPO = "Qwen/Qwen3-0.6B-MLX-8bit"


def _projector_needs_converted_decoder(repo_or_path: str) -> bool:
    """A converted decoder is needed when the projector checkpoint shipped its
    own fine-tuned LM weights (`freeze_language_model=false`) or LoRA adapters
    (`use_lora=true`). Pure projector-only training can reuse the stock MLX
    decoder."""
    from huggingface_hub import hf_hub_download

    p = Path(repo_or_path)
    if p.is_dir():
        cfg_path = p / "config.json"
    else:
        cfg_path = Path(hf_hub_download(repo_or_path, "config.json"))
    cfg = json.loads(cfg_path.read_text())
    if not cfg.get("freeze_language_model", True):
        return True
    return bool(cfg.get("use_lora", False))


def _resolve_default_decoder(projector: str) -> str:
    """Pick the decoder source for a build-bundle invocation that didn't pass
    --decoder. Caches per-projector under .cache/decoder-mlx/<sanitized-repo>
    and auto-runs convert-decoder for fine-tuned/LoRA checkpoints so the
    bundle's decoder matches the projector it was trained against."""
    if not _projector_needs_converted_decoder(projector):
        return _STOCK_DECODER_REPO

    sanitized = projector.replace("/", "--")
    cache_path = _DEFAULT_DECODER_CACHE / sanitized
    if not cache_path.is_dir():
        from scripts.bundle.convert_decoder import convert_decoder

        typer.echo(f"No cached decoder for {projector}; running convert-decoder...")
        convert_decoder(checkpoint=projector, out_dir=cache_path, q_bits=8)
    return str(cache_path)


@app.command("build-bundle")
def build_bundle_cmd(
    projector: Annotated[
        str,
        typer.Option("--projector", "-p", help="HF repo id of the trained projector checkpoint."),
    ],
    decoder: Annotated[
        Optional[str],
        typer.Option(
            "--decoder",
            "-d",
            help=(
                "Local mlx-lm directory or HF repo id. Defaults to a per-projector "
                f"cache under {_DEFAULT_DECODER_CACHE}/<repo>, auto-populated by "
                f"convert-decoder when the projector was trained with a fine-tuned "
                f"LM or LoRA adapters. Falls back to the stock {_STOCK_DECODER_REPO} "
                f"for projector-only checkpoints."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Where to write the bundle."),
    ] = _DEFAULT_BUNDLE_DIR,
    encoder: Annotated[
        str,
        typer.Option("--encoder", help="HF repo id of the upstream PT encoder."),
    ] = "zai-org/GLM-ASR-Nano-2512",
    q_bits: Annotated[int, typer.Option("--q-bits", help="Bits per encoder weight.")] = 8,
    q_group_size: Annotated[
        int, typer.Option("--q-group-size", help="Encoder quantization group size.")
    ] = 64,
) -> None:
    """Assemble the Swift SDK's MLX bundle from projector + decoder + upstream encoder."""
    from scripts.bundle.build_bundle import build_bundle

    if decoder is None:
        decoder = _resolve_default_decoder(projector)
        typer.echo(f"Using decoder: {decoder}")

    build_bundle(
        projector_repo=projector,
        encoder_repo=encoder,
        decoder=decoder,
        output_dir=output_dir,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )


@app.command("push-bundle")
def push_bundle_cmd(
    projector: Annotated[
        str,
        typer.Option(
            "--projector",
            "-p",
            help="HF repo id of the projector this bundle was built from (used in the README).",
        ),
    ],
    repo: Annotated[
        str,
        typer.Option("--repo", help="Target HF repo id (model)."),
    ] = "mazesmazes/tiny-audio-swift-bundle",
    bundle_dir: Annotated[
        Path,
        typer.Option("--bundle-dir", help="Local bundle directory to upload."),
    ] = _DEFAULT_BUNDLE_DIR,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Create the repo as private if it doesn't exist (default: public).",
        ),
    ] = False,
    commit_message: Annotated[
        Optional[str],
        typer.Option("--commit-message", help="HF commit message."),
    ] = None,
) -> None:
    """Push the assembled Swift MLX bundle to a HuggingFace model repo."""
    from scripts.bundle.push_bundle import push_bundle

    msg = commit_message or f"Update bundle from {projector}"
    push_bundle(
        bundle_dir=bundle_dir,
        repo=repo,
        projector_repo=projector,
        private=private,
        commit_message=msg,
    )


if __name__ == "__main__":
    app()
