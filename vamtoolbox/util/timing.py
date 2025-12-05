import time


def timing(func):
    """
    Decorator for timing a function
    """

    def wrap(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print(
            "%s function took %0.4f seconds" % (func.__name__, (end_time - start_time))
        )

        return ret

    return wrap
