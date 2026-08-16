import sys
from pathlib import Path

# Ensure src/ is on sys.path for FastAPI CLI auto-discovery
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.main import app
