import importlib
import logging
import logging.config
import os

import yaml

import models as m


def _fmt_filter(record: logging.LogRecord) -> bool:
    record.levelname = "[%s]" % record.levelname
    record.name = "[%s]" % record.name
    return True


def create_logger(name: str) -> logging.Logger:
    _config = os.path.abspath(str(importlib.resources.files(m).joinpath("logging.yaml")))
    global f, config, LOGGER
    with open(_config, "rt") as f:
        config = yaml.safe_load(f.read())
    # Configure the logging module with the config file
    logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    logger.addFilter(_fmt_filter)
    return logger
