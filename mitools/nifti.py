# Utilities operating on nifti-type images of the nilearn and nibabel interfaces.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
# TODO. Test functions
import os
import numpy as np
import nibabel
import nilearn.image as nl_image

from .default import DEFAULT_RESAMPLE_INTERPOLATION_METHOD
from .exception import InvalidImageFormat
from .validation import (
    checkMultiInputTypes,
    checkInputType)


def loadImg(path_to_img: str) -> nibabel.nifti1.Nifti1Image:
    """ Loads an image in nifti format. """
    if not os.path.exists(path_to_img):
        raise FileNotFoundError(f'Path to image {path_to_img} not found.')

    try:
        return nl_image.load_img(path_to_img)
    except Exception as ex:
        raise Exception(f'Error loading the image. Exception {ex}')


def saveImg(img: nibabel.nifti1.Nifti1Image, file: str, overwrite: bool = False):
    """ Save the image in the specified file """
    if not isinstance(img, nibabel.nifti1.Nifti1Image):
        raise OSError(f'Image must be an instance of nibabel.nifti1.Nifti1Image. Provided type {str(type(img))}')
    if os.path.exists(file) and not overwrite:
        raise FileExistsError(
            f'Image {file} already exists. To overwrite an existing image select the parameter "overwrite" as True.')

    img.to_filename(file)


def loadImgFromData(data: np.ndarray, ref_image: str or nibabel.nifti1.Nifti1Image) -> nibabel.nifti1.Nifti1Image:
    """ Creates a new nifti image from a numpy array and a reference image. """
    checkMultiInputTypes(
        ('data', data, [np.ndarray]),
        ('ref_image', ref_image, [str, nibabel.nifti1.Nifti1Image]))

    if isinstance(ref_image, str):
        if not os.path.exists(ref_image):
            raise FileNotFoundError(f'Path to reference image {ref_image} not found.')
        ref_image = loadImg(ref_image)

    return nl_image.new_img_like(ref_niimg=ref_image, data=data)


def resample(img: nibabel.nifti1.Nifti1Image, reference: nibabel.nifti1.Nifti1Image,
             interpolation: str = DEFAULT_RESAMPLE_INTERPOLATION_METHOD) -> nibabel.nifti1.Nifti1Image:
    """ Function that allows to transform the received image to the same number of dimensions of the reference
    image. This function used nilearn.image.resample_img. """
    checkMultiInputTypes(
        ('img', img, [nibabel.nifti1.Nifti1Image]),
        ('reference', reference, [nibabel.nifti1.Nifti1Image]))

    reference_shape = reference.shape[:3] if len(reference.shape) > 3 else reference.shape

    return nl_image.resample_img(img, img.affine, target_shape=reference_shape, interpolation=interpolation)


def smooth(img: nibabel.nifti1.Nifti1Image, fwhm: int or float or np.ndarray) -> nibabel.nifti1.Nifti1Image:
    """ Function that allows to apply a FHWR smooth to the image. """
    checkMultiInputTypes(
        ('img', img, [nibabel.nifti1.Nifti1Image]),
        ('fwhm', fwhm, [int, float, np.ndarray]))

    return nl_image.smooth_img(img, fwhm)


def resampleVoxelSize(img: nibabel.nifti1.Nifti1Image, voxel_size: int or float or tuple,
                      interpolation: str = DEFAULT_RESAMPLE_INTERPOLATION_METHOD) -> nibabel.nifti1.Nifti1Image:
    """ Function to rescale the image voxels to a certain size. """
    checkMultiInputTypes(
        ('img', img, [nibabel.nifti1.Nifti1Image]),
        ('voxel_size', voxel_size, [int, float, tuple, list]))

    voxel_size = (voxel_size, voxel_size, voxel_size) if isinstance(voxel_size, (int, float)) else voxel_size

    return nl_image.resample_img(img=img, target_affine=np.diag(voxel_size), interpolation=interpolation)


def resampleToImg(img: nibabel.nifti1.Nifti1Image, reference: nibabel.nifti1.Nifti1Image,
                  interpolation: str = DEFAULT_RESAMPLE_INTERPOLATION_METHOD) -> nibabel.nifti1.Nifti1Image:
    """ Resample a Nifti-like source image on a target Nifti-like image (no registration is performed: the image should
    already be aligned). This function used nilearn.image.resample_to_img. """
    checkMultiInputTypes(
        ('img', img, [nibabel.nifti1.Nifti1Image]),
        ('reference', reference, [nibabel.nifti1.Nifti1Image]))

    return nl_image.resample_to_img(img, reference, interpolation=interpolation)


def extractImg(img: nibabel.nifti1.Nifti1Image, index: int = 0) -> nibabel.nifti1.Nifti1Image:
    """ Selects an image within a sequence of images. """
    checkInputType('img', img, [nibabel.nifti1.Nifti1Image])

    if len(img.shape) != 4:
        raise InvalidImageFormat(f'Image must contain 4 dimensions. Image shape {img.shape}')
    if index < img.shape[3]:
        raise IndexError(f'Index {index} not match the image shape {img.shape}')

    return nl_image.index_img(img, index)


def changeAffine(img: nibabel.nifti1.Nifti1Image, affine: np.ndarray) -> nibabel.nifti1.Nifti1Image:
    """ Change the affine transformation of the input image. """
    checkMultiInputTypes(
        ('img', img, [nibabel.nifti1.Nifti1Image]),
        ('affine', affine, [np.ndarray]))

    if affine.shape != (4, 4):
        raise TypeError(f'affine must be a 4x4 matrix. Input shape {affine.shape}')

    new_img = nl_image.new_img_like(ref_niimg=img, data=img.get_fdata(), affine=affine)

    return new_img
