# Workspace Guidelines & Persistent Memory

## 1. Git Push Cadence (Every 8 Changes)
- Track all code and documentation edits.
- Whenever **8 changes** accumulate (tracked in `.agents/git_tracker.json`), or when requested by the user:
  1. Run verification test suite: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
  2. Stage and commit verified changes.
  3. Push cleanly to remote: `git push origin main`.
  4. Reset `.agents/git_tracker.json` change counter to 0.

## 2. Code Quality & Security
- Never commit active API keys, session tokens, or unredacted personal identifiers (guarded by `engine/tools/secret_scanner.py`).
- Maintain 100% green status on contract and UI test suites.
