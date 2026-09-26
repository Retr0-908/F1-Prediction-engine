#!/usr/bin/env bash
set -e
echo "Running Multi-Season Backtest..."
python -m engine.analysis.backtest "$@"
