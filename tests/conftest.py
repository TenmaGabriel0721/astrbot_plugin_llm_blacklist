import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# SDK imports initialize logging and storage before test fixtures run.
_runtime_directory = TemporaryDirectory(prefix="llm-blacklist-tests-")
os.environ.setdefault("ASTRBOT_ROOT", _runtime_directory.name)

# AstrBot loads plugins as packages below its configured plugin directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
