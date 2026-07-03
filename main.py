#!/usr/bin/env python
"""Launch AeroSurvey Studio.

    python main.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aerosurvey.app import main

if __name__ == "__main__":
    raise SystemExit(main())
