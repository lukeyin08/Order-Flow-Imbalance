"""Make the project importable as `src.*` when running pytest from the repo root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
