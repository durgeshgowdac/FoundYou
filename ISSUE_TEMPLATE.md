---
name: Bug report
about: Something isn't working
---

## Description

<!-- A clear, one-paragraph summary of the problem. -->

## Steps to reproduce

```bash
# Exact command you ran
python main.py --...
```

1.
2.
3.

## Expected behaviour

<!-- What you expected to happen. -->

## Actual behaviour

<!-- What actually happened. Paste the full traceback or log output below. -->

```
# log / traceback here
```

## Environment

| Field | Value |
|-------|-------|
| OS | <!-- e.g. macOS 14.4, Ubuntu 22.04, Windows 11 --> |
| Python | <!-- python --version --> |
| PyTorch | <!-- python -c "import torch; print(torch.__version__)" --> |
| Device | <!-- cpu / cuda / mps --> |
| Camera count | <!-- how many cameras connected --> |
| YOLO model | <!-- e.g. yolo11s.pt --> |

## Config overrides

<!-- List any non-default Config values or CLI flags you're using.
     Leave blank if you're running with all defaults. -->

```
--cross-cam-dist ...
--inactive-ttl ...
```

## Additional context

<!-- Screenshots, sample video clips, anything else useful. -->
