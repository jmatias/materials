import logging
import logging.config

import yaml

from fm4m.path_utils import get_path_from_root


def _fmt_filter(record: logging.LogRecord) -> bool:
    record.levelname = "[%s]" % record.levelname
    record.name = "[%s]" % record.name
    return True


def create_logger(name: str) -> logging.Logger:
    _config = get_path_from_root("fm4m.logger", "logging.yaml")
    global f, config, LOGGER
    with open(_config, "rt") as f:
        config = yaml.safe_load(f.read())
    # Configure the logging module with the config file
    logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    logger.addFilter(_fmt_filter)
    return logger
