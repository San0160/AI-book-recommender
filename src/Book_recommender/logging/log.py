import logging
import os
import sys

from Book_recommender.constant import PROJECT_ROOT

LOGGING_STR = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

handlers = [logging.StreamHandler(sys.stdout)]

# Containers should log to stdout only, so the file handler is switchable
if os.environ.get("LOG_TO_FILE", "1") == "1":
    try:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "running.log"))
    except OSError:
        pass  # read-only filesystem: stdout is enough

logging.basicConfig(level=logging.INFO, format=LOGGING_STR, handlers=handlers)
logger = logging.getLogger("Book Recommender logger")