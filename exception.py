import logging
logger = logging.getLogger(__name__)

try:
     error
except Exception as e:
   error_line = e.__traceback__.tb_lineno
   logger.error(traceback.format_exc())
