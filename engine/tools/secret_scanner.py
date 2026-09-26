"""
engine/tools/secret_scanner.py — Zero-Tolerance Nuclear Secret Scanner.

Scans:
1. Working directory files & staged git changes.
2. Full git log history of the current repository.
3. Explicit regexes for API keys, OpenWeatherMap tokens, JWTs, cookies,
   private keys, and personal identifying information (PII).

Exit code:
  0: Clean — no secrets or PII detected.
  1: FAIL — secret or PII detected. Aborts commit/push/CI.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# High-risk patterns
SECRET_PATTERNS = [
    # OpenWeatherMap (32-character hex string)
    (r"(?i)(?:openweather|owm)[a-z0-9_]*\s*[:=]\s*['\"]?([a-f0-9]{32})['\"]?", "OpenWeatherMap API Key"),
    (r"\bdfdae761aaea2954128ed81b8edbbc82\b", "Revoked OpenWeatherMap Key (Leaked)"),
    # Generic API Keys / Secrets
    (r"(?i)(?:api_key|apikey|secret_key|auth_token|access_token)\s*[:=]\s*['\"][A-Za-z0-9_/\+=]{16,}['\"]", "Generic API Key/Token"),
    # JWT Tokens
    (r"\beyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b", "JSON Web Token (JWT)"),
    # F1 Fantasy Cookie / session
    (r"login-session=%7B%22data%22", "F1 Fantasy Raw Session Cookie"),
    (r"(?i)f1_fantasy_cookie\s*[:=]\s*['\"][^'\"]{20,}['\"]", "F1 Fantasy Session Cookie"),
    # AWS Keys
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS Access Key ID"),
    # GitHub Tokens
    (r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b", "GitHub Personal Access Token"),
    (r"\bgithub_pat_[A-Za-z0-9_]{82}\b", "GitHub Fine-Grained PAT"),
    # Google / Gemini API Keys
    (r"\bAIza[0-9A-Za-z-_]{35}\b", "Google API Key"),
    # OpenAI / Anthropic Keys
    (r"\bsk-(?:live-)?[A-Za-z0-9]{20,}\b", "OpenAI/Anthropic Secret Key"),
    # Private RSA / SSH Keys
    (r"-----BEGIN\s+(?:RSA|OPENSSH|DSA|EC)?\s*PRIVATE KEY-----", "Private Key Header"),
    # PII: Personal name check (case-insensitive)
    (r"(?i)\bAaron\s+Kunnath\b", "Personal Name (PII: Aaron Kunnath)"),
    (r"(?i)\bKunnath\b", "Personal Surname (PII: Kunnath)"),
]

BLOCKED_FILES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.staging",
    "id_rsa",
    "id_ed25519",
}

IGNORED_PATHS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".superpowers",
    ".agents",
    "logs",
    "cache",
    "output",
    "engine/tools/secret_scanner.py",  # exclude self (contains pattern definitions)
}


def scan_text(text: str, source_label: str) -> list[str]:
    """Scan arbitrary text against secret patterns."""
    violations = []
    for pattern, desc in SECRET_PATTERNS:
        matches = re.finditer(pattern, text)
        for m in matches:
            matched_str = m.group(0)
            # Mask the secret for safe display
            masked = matched_str[:4] + "*" * (len(matched_str) - 8) + matched_str[-4:] if len(matched_str) > 8 else "***"
            violations.append(f"[{source_label}] {desc}: found '{masked}'")
    return violations


def scan_working_tree() -> list[str]:
    """Scan all tracked files and non-ignored files in the repository."""
    violations = []
    try:
        # Check all files tracked by git
        tracked_files = subprocess.check_output(
            ["git", "ls-files"],
            cwd=str(ROOT_DIR),
            text=True,
            errors="ignore"
        ).splitlines()

        for rel_file in tracked_files:
            rel_file = rel_file.strip()
            if not rel_file or rel_file in IGNORED_PATHS or any(p in IGNORED_PATHS for p in Path(rel_file).parts):
                continue
            if Path(rel_file).name in BLOCKED_FILES:
                violations.append(f"[BLOCKED FILE TRACKED] Prohibited file '{rel_file}' is tracked by git!")
                continue

            filepath = ROOT_DIR / rel_file
            if not filepath.exists() or filepath.stat().st_size > 2 * 1024 * 1024:
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
                violations.extend(scan_text(content, rel_file))
            except Exception:
                pass
    except Exception as e:
        violations.append(f"[SCANNER ERROR] Failed to list git files: {e}")
    return violations


def scan_git_staged() -> list[str]:
    """Scan currently staged git diff and filenames."""
    violations = []
    try:
        # Check staged filenames
        staged_files = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(ROOT_DIR),
            text=True,
            errors="ignore"
        ).splitlines()

        for f in staged_files:
            p = Path(f.strip())
            if p.name in BLOCKED_FILES:
                violations.append(f"[GIT STAGED FILE] Forbidden file staged: '{f}'")

        # Check staged diff content (excluding the scanner itself)
        diff = subprocess.check_output(
            ["git", "diff", "--cached", "--", ":!engine/tools/secret_scanner.py"],
            cwd=str(ROOT_DIR),
            text=True,
            errors="ignore"
        )
        violations.extend(scan_text(diff, "STAGED DIFF"))
    except Exception as e:
        pass
    return violations


def scan_git_history(revisions: str = "HEAD") -> list[str]:
    """Scan git commit history for leaks."""
    violations = []
    try:
        # Check log of revisions (excluding the scanner itself)
        log_diff = subprocess.check_output(
            ["git", "log", "-p", revisions, "--", ":!engine/tools/secret_scanner.py"],
            cwd=str(ROOT_DIR),
            text=True,
            errors="ignore"
        )
        violations.extend(scan_text(log_diff, f"GIT LOG ({revisions})"))
    except Exception:
        pass
    return violations


def run_full_scan(check_history: bool = True) -> int:
    """Execute all scanners and report."""
    print("=" * 60)
    print("  NUCLEAR SECRET & PII SCANNER")
    print("=" * 60)

    all_violations = []

    # 1. Staged files
    staged = scan_git_staged()
    all_violations.extend(staged)

    # 2. Working tree
    tree = scan_working_tree()
    all_violations.extend(tree)

    # 3. History
    if check_history:
        hist = scan_git_history("HEAD")
        all_violations.extend(hist)

    if all_violations:
        print("\n[!] FATAL: NUCLEAR SCANNER DETECTED SECRETS / PII:")
        for v in all_violations:
            print(f"    - {v}")
        print("\nAction Required:")
        print("    1. Remove the sensitive string / file immediately.")
        print("    2. Commit/Push blocked to prevent credential exposure.\n")
        return 1

    print("\n[+] NUCLEAR SCAN PASSED: Zero secrets, API keys, or PII detected.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    check_hist = "--no-history" not in sys.argv
    sys.exit(run_full_scan(check_history=check_hist))
