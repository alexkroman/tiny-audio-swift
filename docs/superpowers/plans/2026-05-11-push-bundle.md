# push-bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ta mlx push-bundle` CLI command that uploads the assembled Swift MLX bundle to a HuggingFace model repo with an auto-generated README.

**Architecture:** New module `scripts/bundle/push_bundle.py` exporting `push_bundle()` plus a pure `_render_readme()` helper. The CLI command in `scripts/bundle/cli.py` is a thin wrapper that builds defaults and calls into the module — matches the existing pattern used by `build_bundle` and `convert_decoder`.

**Tech Stack:** Python, Typer (CLI), `huggingface_hub` (already in `pyproject.toml`).

**Spec:** `docs/superpowers/specs/2026-05-11-push-bundle-design.md`

**No test suite:** This build pipeline has no Python tests (parity tests were dropped in commit `16868b7`). The spec explicitly says no tests for `push_bundle`. Verification is via manual runs documented in each task.

---

## File Structure

- **Create** `scripts/bundle/push_bundle.py` — `push_bundle()` orchestration + `_render_readme()` helper.
- **Modify** `scripts/bundle/cli.py` — add `push_bundle_cmd` Typer command.

---

### Task 1: Create `push_bundle.py` with `_render_readme`

**Files:**
- Create: `scripts/bundle/push_bundle.py`

- [ ] **Step 1: Create the module with `_render_readme`**

Write `scripts/bundle/push_bundle.py`:

```python
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
```

- [ ] **Step 2: Smoke-test `_render_readme` against the existing bundle**

Run:

```bash
cd /Users/alexkroman/Code/tiny-audio-swift
python -c "
from pathlib import Path
from scripts.bundle.push_bundle import _render_readme
print(_render_readme(Path('swift/Sources/TinyAudio/Resources/Model'), 'mazesmazes/tiny-audio-embedded'))
"
```

Expected: A readable README prints to stdout, with the projector link, quant bits/group, encoder dims, audio token, hop length, and format version filled in from `config.json`. No exceptions.

If `swift/Sources/TinyAudio/Resources/Model/config.json` doesn't exist on disk yet, run `ta mlx build-bundle --projector mazesmazes/tiny-audio-embedded` first, or skip this step and verify in Task 4.

- [ ] **Step 3: Commit**

```bash
git add scripts/bundle/push_bundle.py
git commit -m "Add _render_readme helper for push-bundle"
```

---

### Task 2: Add `push_bundle` orchestration

**Files:**
- Modify: `scripts/bundle/push_bundle.py`

- [ ] **Step 1: Add the `push_bundle` function**

Append to `scripts/bundle/push_bundle.py` (after `_render_readme`):

```python
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
        raise SystemExit(
            f"{bundle_dir} has no manifest.json — run `ta mlx build-bundle` first."
        )

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
```

- [ ] **Step 2: Verify the module imports cleanly**

Run:

```bash
cd /Users/alexkroman/Code/tiny-audio-swift
python -c "from scripts.bundle.push_bundle import push_bundle, _render_readme; print('ok')"
```

Expected: prints `ok`. No `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add scripts/bundle/push_bundle.py
git commit -m "Add push_bundle orchestration (validate, README, create_repo, upload)"
```

---

### Task 3: Wire up `push-bundle` Typer command

**Files:**
- Modify: `scripts/bundle/cli.py`

- [ ] **Step 1: Add the `push_bundle_cmd` Typer command**

Add this command to `scripts/bundle/cli.py`, placed after `build_bundle_cmd` (so it appears below `build-bundle` in `--help`):

```python
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
```

Note: `Annotated`, `Optional`, `Path`, `typer`, `_DEFAULT_BUNDLE_DIR`, and `app` are all already imported / defined at the top of `cli.py`. No new imports needed at module top — the local `from scripts.bundle.push_bundle import push_bundle` matches the pattern used by `convert_decoder_cmd` and `build_bundle_cmd`.

- [ ] **Step 2: Verify the command is registered**

Run:

```bash
cd /Users/alexkroman/Code/tiny-audio-swift
python -m scripts.bundle.cli --help
```

Expected: output lists three commands — `convert-decoder`, `build-bundle`, `push-bundle`.

Then:

```bash
python -m scripts.bundle.cli push-bundle --help
```

Expected: shows the `push-bundle` options — `--projector` (required), `--repo` (default `mazesmazes/tiny-audio-swift-bundle`), `--bundle-dir` (default the resources path), `--private` (flag), `--commit-message`.

- [ ] **Step 3: Commit**

```bash
git add scripts/bundle/cli.py
git commit -m "Wire up push-bundle CLI command"
```

---

### Task 4: End-to-end smoke test

**Files:** none modified — manual verification only.

- [ ] **Step 1: Ensure an HF token is available**

Run:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
```

Expected: prints your HF username. If it errors with auth, run `huggingface-cli login` first, or set `HF_TOKEN` in the environment.

- [ ] **Step 2: Confirm a built bundle exists**

Run:

```bash
ls swift/Sources/TinyAudio/Resources/Model/manifest.json
```

Expected: file exists. If not, run `ta mlx build-bundle --projector mazesmazes/tiny-audio-embedded` first.

- [ ] **Step 3: Push to a throwaway test repo**

Pick a test repo id under your account, e.g. `<your-username>/tiny-audio-swift-bundle-test`. Run:

```bash
python -m scripts.bundle.cli push-bundle \
    --projector mazesmazes/tiny-audio-embedded \
    --repo <your-username>/tiny-audio-swift-bundle-test
```

Expected:
- Command completes without error.
- Final line prints `Pushed to https://huggingface.co/<your-username>/tiny-audio-swift-bundle-test`.
- `swift/Sources/TinyAudio/Resources/Model/README.md` now exists locally.

- [ ] **Step 4: Inspect the HF repo in a browser**

Open `https://huggingface.co/<your-username>/tiny-audio-swift-bundle-test`. Verify:
- All bundle files present (`encoder.safetensors`, `projector.safetensors`, `decoder.safetensors`, `config.json`, `decoder_config.json`, `tokenizer.json`, `tokenizer_config.json`, `manifest.json`, `README.md`).
- README renders with the projector link, quant info, encoder dims, audio token.
- Repo is public (not private), since `--private` was not passed.

- [ ] **Step 5: Re-run to confirm idempotency**

Run the same command again:

```bash
python -m scripts.bundle.cli push-bundle \
    --projector mazesmazes/tiny-audio-embedded \
    --repo <your-username>/tiny-audio-swift-bundle-test
```

Expected:
- Command completes without error (no "repo already exists" failure — `exist_ok=True` handles that).
- A new commit appears on the HF repo (may be empty if nothing changed, or contain the README diff).

- [ ] **Step 6: (Optional) Clean up the test repo**

If you want to delete the test repo, do it from the HF web UI (`Settings → Delete this model`). Leaving it is also fine.

- [ ] **Step 7: Commit the locally-generated README**

The README was written to `swift/Sources/TinyAudio/Resources/Model/README.md` during the smoke test. Commit it so the local bundle dir matches what's on HF:

```bash
git add swift/Sources/TinyAudio/Resources/Model/README.md
git commit -m "Add generated bundle README"
```

If the bundle dir is gitignored or you prefer not to track the README, skip this step.
