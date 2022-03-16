# Module with common utilities
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
from copy import deepcopy


def copy(obj: object) -> object:
    """ Performs a deep copy of the input object. """
    return deepcopy(obj)
