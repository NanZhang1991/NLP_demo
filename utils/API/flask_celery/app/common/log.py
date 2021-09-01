import logging
from logging import handlers 
from concurrent_log_handler import ConcurrentRotatingFileHandler
import os
import sys

def logger(filename, module_name):
    logging.basicConfig()
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)    
    th = ConcurrentRotatingFileHandler(filename=filename, mode='a', maxBytes=10 * 1000000,
                                                               backupCount=10, use_gzip=False,
                                                               encoding='utf-8')
    th.suffix = "%Y-%m-%d.log"
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    th.setFormatter(formatter)
    th.setLevel(logging.DEBUG)
    logger.addHandler(th)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.removeHandler(stream_handler)
    return logger


if __name__ == "__main__":
    log_dir = os.path.join(os.getcwd(), 'doc_review/data/log_output')
    fn = 'log.log'
    logger = logger(os.path.join(log_dir, fn), __name__)
    
    logger.debug('Quick zephyrs blow, vexing daft Jim.')
    logger.info('How quickly daft jumping zebras vex.')
    logger.error('unknow')
    print(os.path.basename(sys.argv[0]))