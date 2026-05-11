# push-bundle: publish the Swift MLX bundle to HuggingFace

**Status:** approved (design)
**Date:** 2026-05-11

## Goal

Add a CLI command that uploads the assembled Swift MLX bundle (encoder +
projector + decoder + configs + manifest) to a HuggingFace model repo, so a
trained tiny-audio checkpoint can be published as a ready-to-consume artifact
for `tiny-audio-swift`.

## Non-goals

- Pushing the intermediate converted decoder (`convert-decoder` output).
- Recording build provenance in a new sidecar file (`build_info.json`).
- Automating the push from `build-bundle`; this is an explicit, separate step.

## CLI

New subcommand on `scripts.bundle.cli`:

```
ta mlx push-bundle \
    --projector <hf-repo>            # required; source projector, used in README
    [--repo <hf-repo>]               # default: mazesmazes/tiny-audio-swift-bundle
    [--bundle-dir <path>]            # default: swift/Sources/TinyAudio/Resources/Model
    [--private]                      # default: public
    [--commit-message <str>]         # default: "Update bundle from <projector>"
```

`--bundle-dir` shares its default with `build-bundle`'s `--output-dir`
(`_DEFAULT_BUNDLE_DIR` in `cli.py`).

## Module layout

One new file: `scripts/bundle/push_bundle.py`.

- `push_bundle(*, bundle_dir, repo, projector_repo, private, commit_message)` —
  main entrypoint, called from the CLI command.
- `_render_readme(bundle_dir, projector_repo) -> str` — pure function that
  generates the README from `bundle_dir/config.json` + `projector_repo`.

The CLI command in `cli.py` is a thin wrapper that imports and calls
`push_bundle`, matching the existing pattern for `convert_decoder` and
`build_bundle`.

## Push behavior

1. Validate `bundle_dir` exists and contains `manifest.json`. Fail fast with a
   clear message if not (so users get a helpful error when they forget to run
   `build-bundle` first).
2. Render the README and write to `bundle_dir/README.md`. Overwritten on each
   push. Persisting (rather than uploading from a temp dir) keeps the local
   bundle matching what's on HF.
3. `HfApi().create_repo(repo, repo_type="model", private=private, exist_ok=True)` —
   creates the repo if missing (public by default), no-op if it already exists.
4. `HfApi().upload_folder(folder_path=bundle_dir, repo_id=repo, commit_message=...)` —
   uploads all files.
5. Print the final URL (`https://huggingface.co/<repo>`).

### Authentication

Rely on `huggingface_hub`'s built-in token resolution — env `HF_TOKEN`, or the
credentials stored by `huggingface-cli login`. No custom token handling. If
unauthenticated, `HfApi` raises a clear error.

### Idempotency / staleness

`upload_folder` overwrites existing files but does not delete remote files that
are missing locally. The bundle file set is stable, and accidentally retaining
a stale file is much less bad than accidentally deleting one. No
`delete_patterns` arg is used.

## README content

Generated from `bundle_dir/config.json` + `projector_repo`. Approximate shape:

```markdown
# tiny-audio Swift bundle

MLX bundle for [tiny-audio-swift](https://github.com/alexkroman/tiny-audio-swift),
built from projector [`<projector_repo>`](https://huggingface.co/<projector_repo>).

## Contents
- `encoder.safetensors` — GLM-ASR encoder, quantized <bits>-bit (group <group>)
- `projector.safetensors` — fp16 projector
- `decoder.safetensors` — MLX-LM decoder
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `decoder_config.json`
- `manifest.json` — sha256 + sizes for all files

## Model config
- Encoder dim: <encoder_dim>, layers: <num_layers>, heads: <num_heads>
- Audio token: `<audio_token>` (id <audio_token_id>)
- Hop length: <hop_length>
- Format version: <mlx_format_version>

## Usage
Loaded automatically by tiny-audio-swift's `Transcriber`.
```

Quant bits / group come from `config.json["encoder"]["quantization"]` (already
populated by `build-bundle`). Encoder dims/layers/heads come from
`config.json["encoder"]`. Audio token, hop length, and format version come from
top-level `config.json` keys.

## Dependencies

`huggingface_hub` is already in `pyproject.toml`. No new dependencies.

## Testing

The build pipeline has no test suite (parity tests were dropped in commit
`16868b7`). No tests will be added for `push_bundle` either. Manual
verification path:

1. Run `ta mlx push-bundle --projector mazesmazes/tiny-audio-embedded --repo <test-repo>`
   against a throwaway HF repo.
2. Confirm files appear on HF, README renders.
3. Re-run to confirm idempotency (overwrites cleanly, no errors).
