# Module with operations related to ROIs (Regions of Interest)
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import nilearn as nl
import nibabel
import pandas as pd
import numpy as np
import multiprocessing as mp
import joblib
import warnings
from typing import List

from .nifti import loadImg
from .decorator import ignore_warning
from .validation import (
    checkMultiInputTypes,
    checkInputType)
from .image import (
    NiftiContainer,
    CachedNiftiContainer,
    resampleToImg)

# Available atlases for ROIs extraction
ATLASES = {
    'aal': nl.datasets.fetch_atlas_aal,
    'destrieux': nl.datasets.fetch_atlas_destrieux_2009
}
# Available aggregations for extracting ROIs values
AVAILABLE_AGGREGATIONS = [
    'sum', 'mean', 'median', 'mininum', 'maximum', 'variance', 'std']


class ReferenceROI(object):
    """
    Class containing sets of regions that are usually aggregated together as ROIs.

    ROIs for AAL atlas:
        - ReferenceROI.aal_cerebellum
            Contains all regions of the cerebellum of the AAL (Automated Anatomical Labelling) atlas.
    """
    aal_cerebellum = [
        'cerebelum_crus1_l', 'cerebelum_crus1_r', 'cerebelum_crus2_l', 'cerebelum_crus2_r', 'cerebelum_3_l',
        'cerebelum_3_r', 'cerebelum_4_5_l', 'cerebelum_4_5_r', 'cerebelum_6_l', 'cerebelum_6_r', 'cerebelum_7b_l',
        'cerebelum_7b_r', 'cerebelum_8_l', 'cerebelum_8_r', 'cerebelum_9_l', 'cerebelum_9_r', 'cerebelum_10_l',
        'cerebelum_10_r'
    ]


@ignore_warning
def extractROIs(
        images: List[nibabel.nifti1.Nifti1Image or NiftiContainer] or nibabel.nifti1.Nifti1Image or NiftiContainer,
        atlas: str, aggregate: str or list = 'mean', atlas_kw: dict = None, n_jobs: int = 1) -> pd.DataFrame:
    """ Function that allows to extract the ROI values in a pandas DataFrame from the input images. """
    def __worker__(_img, _masker, _n):
        if isinstance(_img, NiftiContainer):
            _img = _img.nifti_img
        return _n, _masker.fit_transform(_img).squeeze(0)

    checkMultiInputTypes(
        ('images',    images,    [list, nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('atlas',     atlas,     [str]),
        ('aggregate', aggregate, [str, list]),
        ('atlas_kw',  atlas_kw,  [dict, type(None)]),
        ('n_jobs',    n_jobs,    [int]))

    if atlas not in ATLASES:
        raise TypeError(f'Atlas {atlas} not found. Available atlases are {list(ATLASES.keys())}')

    atlas_kw = {} if atlas_kw is None else atlas_kw
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

    if n_jobs <= 0:
        raise TypeError('n_jobs must be greater than 0 or -1.')

    atlas_loader = ATLASES[atlas](**atlas_kw)
    aggregate = [aggregate] if not isinstance(aggregate, list) else aggregate
    images = [images] if not isinstance(images, list) else images

    for n, img in enumerate(images):
        checkInputType(f'images[{n}]', img, [nibabel.nifti1.Nifti1Image, NiftiContainer])

    dfs = []
    for agg in aggregate:
        if agg not in AVAILABLE_AGGREGATIONS:
            raise TypeError(
                f'Aggregation strategy {agg} not found. Available strategies are {AVAILABLE_AGGREGATIONS}')

        masker = nl.input_data.NiftiLabelsMasker(
            labels_img=atlas_loader.maps, standardize=False, strategy='standard_deviation' if agg == 'std' else agg)

        if n_jobs == 1:
            roi_values = dict(__worker__(img, masker, n) for n, img in enumerate(images))
        else:
            roi_values = dict(
                joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                    joblib.delayed(__worker__)(img, masker, n) for n, img in enumerate(images)))
        roi_values_df = pd.DataFrame(roi_values).T
        return roi_values_df
        # Add prefix when several aggregations are provided
        if len(aggregate) == 1:
            roi_values_df.columns = [label.lower() for label in atlas_loader.labels]
        else:
            roi_values_df.columns = ['%s_%s' % (agg, label.lower()) for label in atlas_loader.labels]

        dfs.append(roi_values_df)

    return pd.concat(dfs, axis=1)


def getAtlasNumVoxels(atlas: str, atlas_kw: dict = None) -> pd.DataFrame:
    """ Returns the number of voxels in each ROI for a given atlas. """
    checkMultiInputTypes(
        ('atlas',     atlas,     [str]),
        ('atlas_kw',  atlas_kw,  [dict, type(None)]))

    atlas_kw = {} if atlas_kw is None else atlas_kw

    if atlas not in ATLASES:
        raise TypeError(f'Atlas {atlas} not found. Available atlases are {list(ATLASES.keys())}')

    if atlas != 'aal':
        warnings.warn(
            'This function has only been tested for the AAL atlas. Unexpected behaviour may occur for other atlases.')
    atlas_loader = ATLASES[atlas](**atlas_kw)
    atlas_data = loadImg(atlas_loader.maps).get_fdata()
    region_id, region_count = np.unique(atlas_data, return_counts=True)
    num_voxels = pd.DataFrame(region_count[1:]).T   # Exclude first region count (not brain regions, checked)
    num_voxels.columns = [label.lower() for label in atlas_loader.labels]

    return num_voxels


def extractROIVoxels(
        images: List[nibabel.nifti1.Nifti1Image or NiftiContainer or CachedNiftiContainer] or nibabel.nifti1.Nifti1Image,
        atlas: str, roi: str, atlas_kw: dict = None, clear_cache: bool = False, n_jobs: int = 1) -> pd.DataFrame:
    """ Function that allows to extract all the metabolism values of a given ROI belonging to the specified brain
    atlas. """
    def __worker__(_img: nibabel.nifti1.Nifti1Image or CachedNiftiContainer or NiftiContainer, _roi_locator: int,
                   _atlas_nifti: nibabel.nifti1.Nifti1Image, _atlas_data: np.ndarray, _n: int, _clear_cache: bool
                   ) -> tuple:
        # Resample image to atlas size (image should be smaller than atlas in all dimensions)
        assert len(_img.shape) == len(_atlas_data.shape), \
            'Images (%d) and atlas (%d) must contain the same number of dimensions.' \
            % (len(_img.shape), len(_atlas_data.shape))

        if isinstance(_img, nibabel.nifti1.Nifti1Image):   # convert nibable.nifti1.NiftiImage to NiftiContainer
            _img = NiftiContainer(_img)

        # Resample image to atlas
        _res_img = resampleToImg(_img, _atlas_nifti, inplace=False)

        # Check shape
        for _i, (_img_dim, _atlas_dim) in enumerate(zip(_res_img.shape, _atlas_data.shape)):
            assert _img_dim == _atlas_dim, \
                'Image (%d) must be equal shape than atlas (%d) along axis %d' % (_img_dim, _atlas_dim, _i)

        _voxel_values = _res_img.data[np.where(_atlas_data == _roi_locator)]  # select voxels associated with the ROI

        # Clear cache (only supported by mitools.image.CachedNiftiContainer instances
        if _clear_cache and isinstance(_img, CachedNiftiContainer):
            _img.clearCache()

        del _res_img

        return _n, _voxel_values

    checkMultiInputTypes(
        ('images',      images,       [list, nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('atlas',       atlas,        [str]),
        ('roi',         roi,          [str]),
        ('n_jobs',      n_jobs,       [int]),
        ('atlas_kw',    atlas_kw,     [dict, type(None)]),
        ('clear_cache', clear_cache,  [bool]))

    if atlas not in ATLASES:
        raise TypeError(f'Atlas {atlas} not found. Available atlases are {list(ATLASES.keys())}')

    if atlas != 'aal':
        warnings.warn(
            'This function has only been tested for the AAL atlas. Unexpected behaviour may occur for other atlases.')

    atlas_kw = {} if atlas_kw is None else atlas_kw
    images = [images] if not isinstance(images, list) else images
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

    if n_jobs <= 0:
        raise TypeError('n_jobs must be greater than 0 or -1.')

    for n, img in enumerate(images):
        checkInputType(f'images[{n}]', img, [nibabel.nifti1.Nifti1Image, NiftiContainer])

    # Get atlas ROIs
    roi = roi.lower()
    atlas_loader = ATLASES[atlas](**atlas_kw)
    atlas_rois = [aroi.lower() for aroi in atlas_loader.labels]

    # Check if ROI is in the atlas
    if roi not in atlas_rois:
        raise TypeError(f'ROI {roi} not found in atlas {atlas} ROIs ({atlas_rois})')

    atlas_nifti = loadImg(atlas_loader.maps)
    atlas_data = np.array(atlas_nifti.get_fdata()).astype(int)
    unique_atlas_values = np.unique(atlas_data)
    assert len(unique_atlas_values) > len(atlas_rois), \
        'Unique  (%d) values greater or equal than the number of rois (%d).' \
        % (len(unique_atlas_values), len(atlas_rois))

    # +1 because 0 index correspond to out of the brain voxels
    roi_locator = np.unique(atlas_data)[atlas_rois.index(roi) + 1]

    if n_jobs == 1:
        df = dict([
            __worker__(_img=img, _roi_locator=roi_locator, _atlas_nifti=atlas_nifti, _atlas_data=atlas_data,
                       _n=n, _clear_cache=clear_cache)
            for n, img in enumerate(images)])
    else:
        df = dict(
            joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                joblib.delayed(__worker__)(
                    _img=img, _roi_locator=roi_locator, _atlas_nifti=atlas_nifti, _atlas_data=atlas_data,
                    _n=n, _clear_cache=clear_cache) for n, img in enumerate(images)))
    df = pd.DataFrame(df).T
    df.columns = ['%s_%d' % (roi, n) for n in range(len(df.columns))]

    return df

