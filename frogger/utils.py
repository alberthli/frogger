import threading
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
        The number of seconds to wait before raising a TimeoutError.

    Returns
    -------
    callable
        A function that can be used to decorate other functions with a timeout.
    """
    # raise warning, not error, if timeout <= 0
    if seconds <= 0:
        raise Warning(
            "The specified timeout index is <= 0, so the function will immediately timeout!"
        )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [
                TimeoutError(
                    f"Function '{func.__name__}' timed out after {seconds} seconds"
                )
            ]
            timer = threading.Timer(
                seconds,
                lambda: result.append(
                    TimeoutError(
                        f"Function '{func.__name__}' timed out after {seconds} seconds"
                    )
                ),
            )

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            timer.start()
            thread.join(seconds)
            timer.cancel()

            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator
