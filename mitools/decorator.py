# Decorators module
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import warnings
import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions as deprecated. It will result in a warning
    being emitted when the function is used."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return wrapper


def ignore_warning(func):
    """ Ignore a given warning occurring during method execution. """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter(action='ignore')
            return func(*args, **kwargs)

    return wrapper

