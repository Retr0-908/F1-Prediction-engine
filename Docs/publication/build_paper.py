#!/usr/bin/env python3
"""
build_paper.py — Automated LaTeX Monograph Compiler.
Executes standard two-pass pdflatex + bibtex compilation pipeline.
"""
import subprocess
import sys
from pathlib import Path

PUB_DIR = Path(__file__).resolve().parent
TEX_FILE = "f1_prediction_engine_monograph.tex"
JOB_NAME = "f1_prediction_engine_monograph"


def run_command(cmd: list[str], description: str) -> None:
    print(f"\n[+] Running: {description} ({' '.join(cmd)})...")
    res = subprocess.run(cmd, cwd=str(PUB_DIR), capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        print(f"[!] ERROR during {description} (exit code {res.returncode}):")
        print("=" * 60)
        # Print tail of stdout/stderr for diagnostics
        stdout_lines = res.stdout.splitlines()
        stderr_lines = res.stderr.splitlines()
        for line in stdout_lines[-40:]:
            print(f"STDOUT: {line}")
        for line in stderr_lines[-40:]:
            print(f"STDERR: {line}")
        print("=" * 60)
        sys.exit(res.returncode)
    print(f"[OK] {description} passed.")


def main():
    print("=" * 60)
    print("  FORMULA 1 PREDICTION ENGINE MONOGRAPH BUILDER")
    print("=" * 60)

    # 1. Pass 1: Generate initial aux file
    run_command(["pdflatex", "-interaction=nonstopmode", TEX_FILE], "Pass 1: Initial pdflatex")

    # 2. BibTeX pass: Process citations
    run_command(["bibtex", JOB_NAME], "Pass 2: BibTeX citation resolver")

    # 3. Pass 2: Ingest citations and aux cross-references
    run_command(["pdflatex", "-interaction=nonstopmode", TEX_FILE], "Pass 3: Cross-reference pdflatex")

    # 4. Pass 3: Final PDF typesetting and geometry alignment
    run_command(["pdflatex", "-interaction=nonstopmode", TEX_FILE], "Pass 4: Final typesetting pdflatex")

    pdf_path = PUB_DIR / f"{JOB_NAME}.pdf"
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"\n[SUCCESS] Monograph compiled successfully: {pdf_path.name} ({size_kb:.1f} KB)")
        print(f"Location: {pdf_path}")
    else:
        print("\n[!] Error: Expected output PDF was not found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
