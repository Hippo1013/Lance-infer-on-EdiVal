#!/usr/bin/env python3
"""Offline scoring CLI. See docs/scoring-protocol.md."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lance_mice.scoring.runner import main
if __name__=='__main__':main()
