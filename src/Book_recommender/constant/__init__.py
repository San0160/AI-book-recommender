# For config path and params path

import os
from pathlib import Path

# src/Book_recommender/constant/__init__.py -> four levels up is the repo root.
# BOOK_RECOMMENDER_HOME overrides it (needed in Docker, where the package may
# be installed into site-packages rather than run from the source tree).


PROJECT_ROOT = Path(
    os.environ.get("BOOK_RECOMMENDER_HOME", Path(__file__).resolve().parents[3])
)

CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"