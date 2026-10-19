# Task 3: Demonstrate the log levels
# Reuses the same logger from app.py, so these lines also go into service.log
import logging
from app import logger

# Show every level for this demo, including DEBUG
logger.setLevel(logging.DEBUG)

logger.debug('detailed diagnostics for developers')
logger.info('normal event - service handled a request')
logger.warning('unexpected input but still handled')
logger.error('a request failed')
logger.critical('service cannot continue')