"""
FoundYou — Shared Logger
=========================
Import `log` from here in every module instead of calling
logging.getLogger("FoundYou") repeatedly.

    from logger import log
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger("FoundYou")
