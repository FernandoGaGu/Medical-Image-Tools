# Module with image containers and their corresponding functions
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import os.path

import nibabel
import gc
import joblib
import numpy as np
import multiprocessing as mp
import nilearn.image as nl_image
from abc import abstractmethod, ABCMeta

from . import nifti
from .util import copy
from .default import DEFAULT_RESAMPLE_INTERPOLATION_METHOD
from .validation import (
    checkMultiInputTypes,
    checkInputType)
from .exception import (
    InterfaceError,
    ImageWithModifications)


class Image(object):
    """ Basic interface representing a medical image from which all image-related subclasses shall inherit."""
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "Image"

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def getData(self) -> np.ndarray:
        """ Property to get the image data as numpy array. """
        raise NotImplementedError

    @property
    def data(self) -> np.ndarray:
        """ Return the image data as numpy array. """
        np_data = self.getData()
        if not isinstance(np_data, np.ndarray):
            raise InterfaceError(call_to='data', interface='Image', output=np_data, valid_types=[np.ndarray])

        return np_data

    @property
    def shape(self) -> tuple:
        """ Return the image shape. """
        return self.getData().shape


class NiftiContainer(Image):
    """ Class that acts as a container for images in the nifti format.

    Example
    -------
    >>> import mitools as mt

    >>> path_to_img = '/media/user/HD/neuroimaging/data/12891.nii'
    >>> nii_obj = mt.NiftiContainer(path_to_img, params={'key-1': 'value_1', 'key_2': 20})
    >>> nii_obj.params     # Access to input params specified by "params" in __init__
    >>> nii_obj.nifti_img  # Access to nifti image
    >>> nii_obj.data       # Access to nifti data as numpy array
    >>> nii_obj.copy()     # Returns a deepcopy
    >>> nii_obj.save(file='./test.nii', overwrite=True)  # Save the image overwriting existing data
    >>> nii_obj.copy()     # Returns a deepcopy of the image

    For more information about the nifti images returned by "nii_obj.nifti_img" see:
        https://nipy.org/nibabel/reference/nibabel.nifti1.html#nibabel.nifti1.Nifti1Image
    """
    def __init__(self, nifti_img: nibabel.nifti1.Nifti1Image or str, params: dict = None):
        checkMultiInputTypes(
            ('nifti_img', nifti_img, [nibabel.nifti1.Nifti1Image, str]),
            ('params', params, [dict, type(None)]))

        if isinstance(nifti_img, str):
            nifti_img = nifti.loadImg(nifti_img)

        self._nifti_img = nifti_img
        self._params = params if params is not None else {}
        self._modifications = []   # save modifications made to the image

    def __repr__(self):
        return 'NiftiContainer'

    def __str__(self):
        return self.__repr__()

    @property
    def params(self) -> dict:
        return self._params

    @property
    def nifti_img(self) -> nibabel.nifti1.Nifti1Image:
        return self._nifti_img

    @property
    def modifications(self) -> list:
        """ Return a list with the image modifications (modifications made by internal module functions). """
        return self._modifications

    def getData(self) -> np.ndarray:
        return np.array(self.nifti_img.get_fdata())

    def save(self, file: str, overwrite: bool = False):
        """ Save the image. """
        nifti.saveImg(self.nifti_img, file, overwrite=overwrite)

    def copy(self):
        """ Returns a deepcopy of the image. """
        return copy(self)

    def _addModification(self, modification: str):
        """ Add modification to the image. """
        assert isinstance(modification, str)
        self._modifications.append(modification)

    @classmethod
    def fromData(cls, data: np.ndarray, ref_image: str or nibabel.nifti1.Nifti1Image):
        """ Class method to create a new nifti image from array data and a reference image. """
        return NiftiContainer(nifti.loadImgFromData(data=data, ref_image=ref_image))

    def newImgFromProcess(self, img: nibabel.nifti1.Nifti1Image, process: str):
        """ Method used to create new containers from the application of a certain process. This allows all original
        modifications to the image to be retained. """
        checkMultiInputTypes(
            ('img', img, [nibabel.nifti1.Nifti1Image]),
            ('process', process, [str]))

        new_img = NiftiContainer(img)
        for modification in self._modifications:
            new_img._addModification(modification)

        new_img._addModification(process)

        return new_img


class CachedNiftiContainer(NiftiContainer):
    """ Class that acts as a container for images in the nifti format with caching.

    Example
    -------
    >>> import mitools as mt

    >>> path_to_img = '/media/user/HD/neuroimaging/data/12891.nii'

    >>> # Specifying the parameter load as False we are avoiding the data to be in cache
    >>> nii_obj_c = mt.CachedNiftiContainer(path_to_img, load=False, params={'id': 32189})
    >>> nii_obj_c.is_cached     # False, the image is not in cache
    >>> nii_obj_c.load()        # Load the image into cache
    >>> nii_obj_c.is_cached     # True, the image is in cache
    >>> nii_obj_c.clearCache()  # Remove the image from the cache

    Note: this class share the same methods and properties as class mitools.image.NiftiContainer.
    """
    def __init__(self, path: str, load: bool = False, params: dict = None):
        checkMultiInputTypes(
            ('path',  path, [str]),
            ('load', load, [bool]),
            ('params', params, [dict, type(None)]))

        if not os.path.exists(path):
            raise OSError(f'Path {path} not found.')

        super(CachedNiftiContainer, self).__init__(nifti_img=nifti.loadImg(path), params=params)

        self._img_path = path   # save path to image

        if not load:  # remove image from cache
            self._nifti_img = None

        self._cached = load

    def __repr__(self):
        return f'CachedNiftiContainer({self._img_path}, cached={self._cached})'

    @property
    def nifti_img(self) -> nibabel.nifti1.Nifti1Image:
        self.load()   # Load from cache
        return self._nifti_img

    @property
    def is_cached(self) -> bool:
        return self._cached

    @property
    def path(self) -> str:
        return self._img_path

    def load(self):
        """ Load the image into the cache modifying the object inplace. """
        if not self._cached:
            self._nifti_img = nifti.loadImg(self._img_path)
            self._cached = True

    def clearCache(self, remove_mods: bool = False):
        """ Removes the image data from the cache. """
        if not remove_mods and len(self._modifications) > 0:
            raise ImageWithModifications(self._modifications)

        # Clear cache
        del self._nifti_img
        gc.collect()
        self._nifti_img = None

        self._cached = False


def updateData(ref_img: NiftiContainer, data: np.ndarray) -> NiftiContainer:
    """ Function that returns a new mitools.image.NiftiContainer by updating the reference image data. This operation
     is not performed inplace. """
    checkMultiInputTypes(
        ('ref_img', ref_img, [NiftiContainer]),
        ('data', data, [np.ndarray]))

    # Convert the numpy array to a nifti-like image
    new_img = nl_image.new_img_like(data=data, ref_niimg=ref_img.nifti_img)

    return ref_img.newImgFromProcess(new_img, 'updateData')


def resample(img: NiftiContainer, reference: nibabel.nifti1.Nifti1Image or NiftiContainer,
             interpolation: str = DEFAULT_RESAMPLE_INTERPOLATION_METHOD, inplace: bool = False) -> NiftiContainer:
    """ Wrapper of mitools.util.nifti.resample() operating on mitools.image.NiftiContainer instances. """
    checkMultiInputTypes(
        ('img',           img,           [NiftiContainer]),
        ('reference',     reference,     [nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('interpolation', interpolation, [str]))

    img = img.copy() if not inplace else img

    # Resample image to reference image using nilearn.image.resample_img
    reference_img = reference.nifti_img if isinstance(reference, NiftiContainer) else reference
    resampled_img = nifti.resample(img.nifti_img, reference_img, interpolation=interpolation)

    return img.newImgFromProcess(resampled_img, 'resample')


def resampleToImg(img: NiftiContainer, reference: nibabel.nifti1.Nifti1Image or NiftiContainer,
                  interpolation: str = DEFAULT_RESAMPLE_INTERPOLATION_METHOD, inplace: bool = False) -> NiftiContainer:
    """ Wrapper of mitools.util.nifti.resampleToImg() operating on mitools.image.NiftiContainer instances. """
    checkMultiInputTypes(
        ('img',           img,           [NiftiContainer]),
        ('reference',     reference,     [nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('interpolation', interpolation, [str]))

    img = img.copy() if not inplace else img

    # Resample image to reference image using nilearn.image.resample_to_img
    reference_img = reference.nifti_img if isinstance(reference, NiftiContainer) else reference
    resampled_img = nifti.resampleToImg(img.nifti_img, reference_img, interpolation=interpolation)

    return img.newImgFromProcess(resampled_img, 'resampleToImg')


def resampleVoxelSize(img: NiftiContainer, voxel_size: int or tuple, inplace: bool = False) -> NiftiContainer:
    """ Wrapper of mitools.util.nifti.resampleVoxelSize() operating on mitools.image.NiftiContainer instances. """
    checkInputType('img', img, [NiftiContainer])

    img = img.copy() if not inplace else img

    # Resample voxels size
    resampled_img = nifti.resampleVoxelSize(img.nifti_img, voxel_size)

    return img.newImgFromProcess(resampled_img, 'resampleVoxelSize')


def mapFunctionToImg(img: list or NiftiContainer, function: callable, n_jobs: int = 1, **kwargs) -> list:
    """ Function allowing to map a function acting at the level of the data of a given image to images. The function
    must take an integer as the first argument and an array of images as the second argument and return the same index
    and a new array of the same shape. The rest of the arguments that the function takes can be defined using kwargs.

    Note that the output will be converted to a dictionary.
    """
    # Check input types
    checkMultiInputTypes(
        ('img', img, [list, NiftiContainer]),
        ('n_jobs', n_jobs, [int]))
    if not callable(function):
        raise TypeError('The "function" argument must be a callable.')

    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
    if n_jobs <= 0:
        raise TypeError('"n_jobs" must be -1 or greater than 0.')

    imgs = [img] if not isinstance(img, list) else img
    for i, img in enumerate(imgs):
        checkInputType(f'img[{i}]', img, [NiftiContainer])

    try:
        if n_jobs == 1:
            output = [function(i, img.data, **kwargs) for i, img in enumerate(imgs)]
        else:
            output = joblib.Parallel(backend='loky', n_jobs=n_jobs)(
                joblib.delayed(function)(i, img.data, **kwargs) for i, img in enumerate(imgs))
    except Exception as ex:
        raise Exception(f'The function could not be computed. Exception thrown: {ex}')

    unique_keys = []
    output_imgs = []
    for n, (index, data) in enumerate(output):
        # Check returned format
        checkMultiInputTypes(
            (f'output[{n}][0]', index, [int]),
            (f'output[{n}][1]', data, [np.ndarray]))

        # Check for duplicated indices
        if index in unique_keys:
            raise IndexError(f'Index {index} is duplicated.')

        unique_keys.append(index)
        output_imgs.append(NiftiContainer.fromData(data=data, ref_image=imgs[index].nifti_img))

    return output_imgs
