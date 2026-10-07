# Contributing to FoundYou

Thanks for taking the time to contribute. This document covers how to report bugs, suggest features, and submit code changes.

---

## Table of contents

- [Reporting bugs](#reporting-bugs)
- [Suggesting features](#suggesting-features)
- [Setting up locally](#setting-up-locally)
- [Making changes](#making-changes)
- [Submitting a pull request](#submitting-a-pull-request)
- [Code style](#code-style)
- [Module responsibilities](#module-responsibilities)

---

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE.md) when opening an issue. The most useful bug reports include:

- A minimal script or command that reproduces the issue
- The full traceback or log output
- Your OS, Python version, and device (`cpu` / `cuda` / `mps`)
- Whether it happens with a single camera or multiple

---

## Suggesting features

Open an issue with the `enhancement` label. Describe the problem you're trying to solve rather than jumping straight to a proposed solution — there may already be a config knob that covers it, or a simpler approach than you'd expect.

---

## Setting up locally

```bash
git clone https://github.com/your-org/foundyou.git
cd foundyou

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install ultralytics torch torchvision faiss-cpu opencv-python
pip install torchreid gdown       # optional but recommended
```

Verify the setup:

```bash
python main.py --debug
```

If you see OSNet initialising and cameras being detected, you're good.

---

## Making changes

1. Fork the repo and create a branch from `main`:

   ```bash
   git checkout -b fix/short-description
   # or
   git checkout -b feat/short-description
   ```

2. Make your changes. Keep commits focused — one logical change per commit.

3. If you're changing matching behaviour (thresholds, the resolve pipeline, the merge pass), add a note to your PR explaining how you tested it and what the before/after effect on false-positive rate looked like.

4. Run a quick sanity check:

   ```bash
   python -c "from reid import GlobalReIDManager; print('imports ok')"
   ```

---

## Submitting a pull request

- Target the `main` branch
- Fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
- Keep PRs small and focused — a 200-line PR gets reviewed faster than a 2000-line one
- If your PR fixes an open issue, reference it: `Closes #42`

---

## Code style

- **Formatting**: PEP 8. Line length 100.
- **Imports**: stdlib → third-party → local, each group separated by a blank line.
- **Type hints**: use them for function signatures. `np.ndarray | None` style (Python 3.10+).
- **Logging**: always use `from logger import log`, never create a new logger.
- **No logic in `main.py`**: the main loop wires modules together; actual logic belongs in the relevant module.

---

## Module responsibilities

Before adding code, check which module it belongs in:

| Module | Owns |
|--------|------|
| `config.py` | Constants and thresholds only — no logic |
| `logger.py` | Single shared logger — do not modify |
| `models.py` | OSNet architecture and weight loading |
| `features.py` | Pure distance/similarity functions |
| `tracks.py` | `GlobalTrack` state and `DB` persistence |
| `reid.py` | Resolve, merge, archive orchestration |
| `camera.py` | Per-camera detection thread |
| `ui.py` | OpenCV drawing only — no ReID logic |
| `main.py` | Wiring and CLI only — no business logic |

If something doesn't fit cleanly, open an issue to discuss before sending a PR.
