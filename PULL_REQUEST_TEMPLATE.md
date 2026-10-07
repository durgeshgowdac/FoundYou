## What does this PR do?

<!-- One or two sentences. What problem does it solve, or what does it add? -->

Closes #<!-- issue number, if applicable -->

---

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behaviour change)
- [ ] Config / threshold tuning
- [ ] Docs / comments only

---

## Changes

<!-- Bullet list of what you changed and why.
     For matching/ReID changes, explain the before/after effect on accuracy. -->

-
-

---

## Testing

<!-- How did you verify this works? e.g.:
     - Tested with 2 cameras, 3 people, ~10 minute session
     - False-positive rate dropped from X to Y
     - Ran python -c "from reid import GlobalReIDManager" with no errors -->

- [ ] Tested with a live camera feed
- [ ] Imports cleanly (`python -c "from reid import GlobalReIDManager"`)
- [ ] No new `foundyou.pkl` / `.faiss` files accidentally committed

---

## Checklist

- [ ] Code follows the module responsibilities in `CONTRIBUTING.md`
- [ ] New logic is in the right module, not in `main.py`
- [ ] Uses `from logger import log`, not a new `getLogger` call
- [ ] No model weights (`*.pt`, `*.pth`) or runtime data (`*.pkl`, `*.faiss`) included
