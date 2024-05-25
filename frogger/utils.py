import signal
from concurrent.futures import TimeoutError
from functools import wraps


def timeout(seconds: float) -> callable:
    """A decorator that can be used to timeout arbitrary functions after specified time.

    Example Usage
    -------------
    @timeout(2)
    def long_running_function():
        time.sleep(5)
        return "Finished"

    try:
        result = long_running_function()
        print(result)
    except TimeoutError as e:
        print(e)

    Alternatively, you can wrap another function with it as follows.
    timed_func = timeout(2)(long_running_function)

    Parameters
    ----------
    seconds : float
        The number of seconds to wait before raising a TimeoutError. If seconds is 0.0,
        then there is NO timeout!

    Returns
    -------
    callable
        A function that can be used to decorate other functions with a timeout.
    """

    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(
                f"Function '{func.__name__}' timed out after {seconds} seconds"
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # set the signal handler and an interval timer
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the timer
                signal.setitimer(signal.ITIMER_REAL, 0)
            return result

        return wrapper

    return decorator
