#!/usr/bin/env python3
"""Explicit entry point for the independent Qwen single-vote protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lance_mice.scoring_qwen_once import main
if __name__ == '__main__':
    main()
