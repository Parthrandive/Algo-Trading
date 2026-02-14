import logging
import sys
import json
import os
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def setup_logger(name: str, level=logging.INFO, json_format=False) -> logging.Logger:
    """
    Sets up a logger with either standard console output or JSON formatting.
    
    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (default INFO)
        json_format: Boolean to enable JSON formatting (default False, can be overridden by env LOG_FORMAT=json)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Check environment variable for override
    if os.getenv("LOG_FORMAT", "").lower() == "json":
        json_format = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if setup prompt is called multiple times
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
