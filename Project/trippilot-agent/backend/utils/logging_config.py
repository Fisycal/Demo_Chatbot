from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(service_name: str = "vacation-agent") -> logging.Logger:
    """Configure application logging.

    Env vars:
      LOG_LEVEL: INFO|DEBUG|WARNING|ERROR (default INFO)
      LOG_FILE: path to a log file (optional). If set, logs also go to a rotating file.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(service_name)
    logger.setLevel(level)

    # Avoid duplicate handlers (e.g., uvicorn reload)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_file = os.getenv("LOG_FILE", "").strip()
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Reduce noisy libraries
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("httpx").setLevel(os.getenv("HTTPX_LOG_LEVEL", "WARNING").upper())

    return logger
