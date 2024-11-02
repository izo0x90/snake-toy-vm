import builtins
import logging
import os

from rich.logging import RichHandler

import config


def init_logging():
    log_path = f"{config.LOG_PREFIX}{os.getpid()}.log"

    # Overriding default print in 3.13 breaks builtin color exception text
    # use `rprint` for formatted print

    builtins.print = print
    logging_level = logging.DEBUG
    logger_config = {
        "handlers": [RichHandler(rich_tracebacks=False, tracebacks_show_locals=True)],  # type: ignore
        "format": "PID(%(process)d) - %(message)s",
        "level": logging_level,
    }
    logging.basicConfig(**logger_config)
    # instantiate logger
    logger = logging.getLogger()

    if config.LOG_TO_FILES:
        handler = logging.FileHandler(log_path, mode="w")
        logger.addHandler(handler)

    logger.info(f"Logging initialized at {logging_level=} for {os.getpid()=} ...")


init_logging()
