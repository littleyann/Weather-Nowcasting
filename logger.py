import logging


def load_logger(log_path, if_save=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s", datefmt="%a %b %d %H:%M:%S %Y")

    s_handler = logging.StreamHandler()
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)

    if if_save:
        f_handler = logging.FileHandler(log_path + '/log.txt', mode='w')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    return logger
