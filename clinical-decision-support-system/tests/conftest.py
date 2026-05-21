"""
Shared pytest configuration.
Adds the project root to sys.path so imports work without installing the package.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
