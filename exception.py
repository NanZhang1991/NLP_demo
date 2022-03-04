import logging
logger = logging.getLogger(__file__)

try:
     error
except Exception as e:
   error_line = e.__traceback__.tb_lineno
   logger.error(traceback.format_exc())
