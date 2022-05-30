import logging


def get_logger(name: str = "default", level=logging.DEBUG):
    # format
    formats = logging.Formatter("[%(name)s | %(levelname)s] %(message)s")

    # stream handler
    handler = logging.StreamHandler()
    handler.setFormatter(formats)

    # logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
