"""Import this module to silence warnings from importlogger"""

import logging

from .importlogger import logger

logger.setLevel(logging.ERROR)
