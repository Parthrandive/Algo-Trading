import time
import functools
import random
from typing import Callable, Any, Type, Tuple

# Define a custom exception if not already available, or use Exception
class RateLimitExceeded(Exception):
    pass

def retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: int = 1,
    exceptions_to_check: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    :param retries: Number of times to retry.
    :param backoff_in_seconds: Initial backoff time in seconds.
    :param exceptions_to_check: Tuple of exceptions to catch and retry upon.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions_to_check as e:
                    if x == retries:
                        raise e
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

def rate_limit(
    calls: int,
    period: int
) -> Callable:
    """
    Rate limiting decorator.
    
    :param calls: Number of allowed calls.
    :param period: Period in seconds.
    """
    interval = period / float(calls)
    
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time.time() - last_called[0]
            left_to_wait = interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            val = func(*args, **kwargs)
            last_called[0] = time.time()
            return val
        return wrapper
    return decorator
