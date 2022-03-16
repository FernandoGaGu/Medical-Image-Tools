# Module with tools related to data validation.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#

def checkInputType(input_var_name: str, input_var: object, valid_types: list or tuple):
    """ Function that checks that the type of the input variable "input_var" is within the valid types "valid_types"."""
    if not isinstance(input_var_name, str):
        raise TypeError(f'input_var_name must be a string. Provided {str(input_var_name)}')
    if not isinstance(valid_types, (tuple, list)):
        raise TypeError(f'valid_types must be a list or a tuple. Provided {str(valid_types)}')

    if not isinstance(input_var, tuple(valid_types)):
        raise TypeError(
            f'Input variable {input_var_name} of type {str(type(input_var))} not in available types: '
            f'{",".join([str(v) for v in valid_types])}.')


def checkMultiInputTypes(*args):
    """ Wrapper of function A to check multiple variables at the same time. """
    for element in args:
        if not ((isinstance(element, tuple) or isinstance(element, list)) and len(element) == 3):
            raise TypeError('The arguments of this function must consist of tuples or lists of three arguments '
                            'following the signature of the adhoc.utils.checkInputType() function.')

        checkInputType(*element)