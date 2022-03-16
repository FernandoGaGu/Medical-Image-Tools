# Exceptions module
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#

class InterfaceError(Exception):
    """ Exception thrown when an interface method call does not return the appropriate type. """
    def __init__(self, call_to: str, interface: str, output: object, valid_types: list):
        super(InterfaceError, self).__init__(
            f'The call to method {call_to} of interface {interface} is of type {str(type(output))} when the valid '
            f'types are {[str(type(tp) for tp in valid_types)]}.')


class ImageWithModifications(Exception):
    """ Exception thrown when NiftiImage contains modifications and someone is trying to remove the image from cache.
    """
    def __init__(self, modifications: list):
        super(ImageWithModifications, self).__init__(
            f'The image to be deleted from memory contains modifications {modifications}. If the cache is cleared the ' 
            'modifications will be lost. To allow the latter specify the parameter remove_mods as True.')


class InvalidImageFormat(Exception):
    """ Exception thrown when there is a problem with the image dimensions. """
    pass


