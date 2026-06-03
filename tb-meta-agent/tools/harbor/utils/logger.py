from __future__ import annotations

import logging


def setup_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        # Keep default handlers; this is a lightweight stub.
        lg.setLevel(logging.INFO)
    return lg


logger = setup_logger(__name__)

