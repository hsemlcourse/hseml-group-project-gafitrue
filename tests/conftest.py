"""Добавляем корень проекта в sys.path, чтобы tests/ видели src/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
