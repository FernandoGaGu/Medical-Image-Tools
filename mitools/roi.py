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
from nilearn.maskers import NiftiLabelsMasker
from tqdm import tqdm
from typing import Tuple

from . import nifti
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

#warnings.filterwarnings("ignore")


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
        'cerebelum_10_r', 'vermis_1_2', 'vermis_3', 'vermis_4_5', 'vermis_6', 'vermis_7', 'vermis_8', 'vermis_9',
        'vermis_10'
    ]


class AALCode(object):
    """ Class with a dictionary used to abbreviate the regions of the AAL atlas.
    Nomenclature from https://www.sciencedirect.com/science/article/pii/S1053811919307803

    """
    alias = {
         'precentral_l': 'PreCG (L)',
         'precentral_r': 'PreCG (R)',
         'frontal_sup_l': 'SFG (L)',
         'frontal_sup_r': 'SFG (R)',
         'frontal_sup_orb_l': 'SFGorb (L)',
         'frontal_sup_orb_r': 'SFGorb (R)',
         'frontal_mid_l': 'MFG (L)',
         'frontal_mid_r': 'MFG (R)',
         'frontal_mid_orb_l': 'MFGorb (L)',
         'frontal_mid_orb_r': 'MFGorb (R)',
         'frontal_inf_oper_l': 'IFGoperc (L)',
         'frontal_inf_oper_r': 'IFGoperc (R)',
         'frontal_inf_tri_l': 'IFGtriang (L)',
         'frontal_inf_tri_r': 'IFGtriang (R)',
         'frontal_inf_orb_l': 'IFGorb (L)',
         'frontal_inf_orb_r': 'IFGorb (R)',
         'rolandic_oper_l': 'ROL (L)',
         'rolandic_oper_r': 'ROL (R)',
         'supp_motor_area_l': 'SMA (L)',
         'supp_motor_area_r': 'SMA (R)',
         'olfactory_l': 'OLF (L)',
         'olfactory_r': 'OLF (R)',
         'frontal_sup_medial_l': 'SFGmedial (L)',
         'frontal_sup_medial_r': 'SFGmedial (R)',
         'frontal_med_orb_l': 'PFCventmed (L)',
         'frontal_med_orb_r': 'PFCventmed (R)',
         'rectus_l': 'REC (L)',
         'rectus_r': 'REC (R)',
         'insula_l': 'INS (L)',
         'insula_r': 'INS (R)',
         'cingulum_ant_l': 'ACC (L)',
         'cingulum_ant_r': 'ACC (R)',
         'cingulum_mid_l': 'MCC (L)',
         'cingulum_mid_r': 'MCC (R)',
         'cingulum_post_l': 'PCC (L)',
         'cingulum_post_r': 'PCC (R)',
         'hippocampus_l': 'HIP (L)',
         'hippocampus_r': 'HIP (R)',
         'parahippocampal_l': 'PHG (L)',
         'parahippocampal_r': 'PHG (R)',
         'amygdala_l': 'AMYG (L)',
         'amygdala_r': 'AMYG (R)',
         'calcarine_l': 'CAL (L)',
         'calcarine_r': 'CAL (R)',
         'cuneus_l': 'CUN (L)',
         'cuneus_r': 'CUN (R)',
         'lingual_l': 'LING (L)',
         'lingual_r': 'LING (R)',
         'occipital_sup_l': 'SOG (L)',
         'occipital_sup_r': 'SOG (R)',
         'occipital_mid_l': 'MOG (L)',
         'occipital_mid_r': 'MOG (R)',
         'occipital_inf_l': 'IOG (L)',
         'occipital_inf_r': 'IOG (R)',
         'fusiform_l': 'FFG (L)',
         'fusiform_r': 'FFG (R)',
         'postcentral_l': 'PoCG (L)',
         'postcentral_r': 'PoCG (R)',
         'parietal_sup_l': 'SPG (L)',
         'parietal_sup_r': 'SPG (R)',
         'parietal_inf_l': 'IPG (L)',
         'parietal_inf_r': 'IPG (R)',
         'supramarginal_l': 'SMG (L)',
         'supramarginal_r': 'SMG (R)',
         'angular_l': 'ANG (L)',
         'angular_r': 'ANG (R)',
         'precuneus_l': 'PCUN (L)',
         'precuneus_r': 'PCUN (R)',
         'paracentral_lobule_l': 'PCL (L)',
         'paracentral_lobule_r': 'PCL (R)',
         'caudate_l': 'CAU (L)',
         'caudate_r': 'CAU (R)',
         'putamen_l': 'PUT (L)',
         'putamen_r': 'PUT (R)',
         'pallidum_l': 'PAL (L)',
         'pallidum_r': 'PAL (R)',
         'thalamus_l': 'THA (L)',
         'thalamus_r': 'THA (R)',
         'heschl_l': 'HES (L)',
         'heschl_r': 'HES (R)',
         'temporal_sup_l': 'STG (L)',
         'temporal_sup_r': 'STG (R)',
         'temporal_pole_sup_l': 'TPOsup (L)',
         'temporal_pole_sup_r': 'TPOsup (R)',
         'temporal_mid_l': 'MTG (L)',
         'temporal_mid_r': 'MTG (R)',
         'temporal_pole_mid_l': 'TPOmid (L)',
         'temporal_pole_mid_r': 'TPOmid (R)',
         'temporal_inf_l': 'ITG (L)',
         'temporal_inf_r': 'ITG (R)',
         'cerebelum_crus1_l': 'CERCRU1 (L)',
         'cerebelum_crus1_r': 'CERCRU1 (R)',
         'cerebelum_crus2_l': 'CERCRU2 (L)',
         'cerebelum_crus2_r': 'CERCRU2 (R)',
         'cerebelum_3_l': 'CER3 (L)',
         'cerebelum_3_r': 'CER3 (R)',
         'cerebelum_4_5_l': 'CER4_5 (L)',
         'cerebelum_4_5_r': 'CER4_5 (R)',
         'cerebelum_6_l': 'CER6 (L)',
         'cerebelum_6_r': 'CER6 (R)',
         'cerebelum_7b_l': 'CER7b (L)',
         'cerebelum_7b_r': 'CER7b (R)',
         'cerebelum_8_l': 'CER8 (L)',
         'cerebelum_8_r': 'CER8 (R)',
         'cerebelum_9_l': 'CER9 (L)',
         'cerebelum_9_r': 'CER9 (R)',
         'cerebelum_10_l': 'CER10 (L)',
         'cerebelum_10_r': 'CER10 (R)',
         'vermis_1_2': 'VER1_2',
         'vermis_3': 'VER3',
         'vermis_4_5': 'VER4_5',
         'vermis_6': 'VER6',
         'vermis_7': 'VER7',
         'vermis_8': 'VER8',
         'vermis_9': 'VER9',
         'vermis_10': 'VER10'}


class Deprecated:
    """ Class with all deprecated functions """
    @classmethod
    @ignore_warning
    def extractROIs(
            cls,
            images: List[nibabel.nifti1.Nifti1Image or NiftiContainer] or nibabel.nifti1.Nifti1Image or NiftiContainer,
            atlas: str,
            aggregate: str or list = 'mean',
            atlas_kw: dict = None,
            n_jobs: int = 1,
            verbose: bool = False) -> pd.DataFrame:
        """ Function that allows to extract the ROI values in a pandas DataFrame from the input images. """
        def __worker__(_img, _ref_img, _masker, _n):
            if isinstance(_img, NiftiContainer):
                _img = _img.nifti_img
            # resample the target image to match the atlas image
            _img = nifti.resampleToImg(_img, _ref_img.nifti_img, interpolation='continuous')
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
        atlas_img = NiftiContainer(atlas_loader.maps)
        aggregate = [aggregate] if not isinstance(aggregate, list) else aggregate
        images = [images] if not isinstance(images, list) else images

        for n, img in enumerate(images):
            checkInputType(f'images[{n}]', img, [nibabel.nifti1.Nifti1Image, NiftiContainer])

        dfs = []
        for agg in aggregate:
            if agg not in AVAILABLE_AGGREGATIONS:
                raise TypeError(
                    f'Aggregation strategy {agg} not found. Available strategies are {AVAILABLE_AGGREGATIONS}')

            masker = NiftiLabelsMasker(
                labels_img=atlas_loader.maps, standardize=False, strategy='standard_deviation' if agg == 'std' else agg)

            iterator = tqdm(enumerate(images), desc='Extracting ROIs...') if verbose else enumerate(images)

            if n_jobs == 1:
                roi_values = dict(__worker__(img, atlas_img, masker, n) for n, img in iterator)
            else:
                roi_values = dict(
                    joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                        joblib.delayed(__worker__)(img, atlas_img, masker, n) for n, img in iterator))
            roi_values_df = pd.DataFrame(roi_values).T.sort_index()
            # Add prefix when several aggregations are provided
            if len(aggregate) == 1:
                roi_values_df.columns = [label.lower() for label in atlas_loader.labels]
            else:
                roi_values_df.columns = ['%s_%s' % (agg, label.lower()) for label in atlas_loader.labels]

            dfs.append(roi_values_df)

        return pd.concat(dfs, axis=1)


@ignore_warning
def extractROIs(
        images: List[nibabel.nifti1.Nifti1Image or NiftiContainer] or nibabel.nifti1.Nifti1Image or NiftiContainer,
        atlas: str,
        aggregate: str or list = 'mean',
        atlas_kw: dict = None,
        n_jobs: int = 1,
        resample_to_atlas: bool = True,
        verbose: bool = False) -> pd.DataFrame:
    """ Function that allows to extract the ROI values in a pandas DataFrame from the input images.
    IMPORTANT: The difference of this function wrt extractROIs, is that it resample the AAL atlas
    to the image shape, instead of resampling the image to the AAL atlas shape
    """
    def __worker__(_img, _ref_img, _agg, _resample_to_atlas, _n):
        if isinstance(_img, NiftiContainer):
            _img = _img.nifti_img
        if isinstance(_ref_img, NiftiContainer):
            _ref_img = _ref_img.nifti_img

        # resample atlas to the image shape
        if _resample_to_atlas:  # resample image to atlas shape
            _img = nifti.resampleToImg(_img, _ref_img, interpolation='nearest')
        else:   # resample atlas to image shape
            _ref_img = nifti.resampleToImg(_ref_img, _img, interpolation='nearest')

        # create masker
        _masker = NiftiLabelsMasker(
            labels_img=_ref_img, standardize=False, strategy='standard_deviation' if _agg == 'std' else _agg
        )
        return _n, _masker.fit_transform(_img).squeeze(0)

    checkMultiInputTypes(
        ('images',    images,    [list, nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('atlas',     atlas,     [str]),
        ('resample_to_atlas',     resample_to_atlas,     [bool]),
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
    atlas_img = NiftiContainer(atlas_loader.maps)
    aggregate = [aggregate] if not isinstance(aggregate, list) else aggregate
    images = [images] if not isinstance(images, list) else images

    for n, img in enumerate(images):
        checkInputType(f'images[{n}]', img, [nibabel.nifti1.Nifti1Image, NiftiContainer])

    dfs = []
    for agg in aggregate:
        if agg not in AVAILABLE_AGGREGATIONS:
            raise TypeError(
                f'Aggregation strategy {agg} not found. Available strategies are {AVAILABLE_AGGREGATIONS}')

        iterator = tqdm(enumerate(images), desc='Extracting ROIs...') if verbose else enumerate(images)

        if n_jobs == 1:
            roi_values = dict(__worker__(img, atlas_img, agg, resample_to_atlas, n) for n, img in iterator)
        else:
            roi_values = dict(
                joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                    joblib.delayed(__worker__)(img, atlas_img, agg, resample_to_atlas, n) for n, img in iterator))
        roi_values_df = pd.DataFrame(roi_values).T.sort_index()
        # Add prefix when several aggregations are provided
        if len(aggregate) == 1:
            roi_values_df.columns = [label.lower() for label in atlas_loader.labels]
        else:
            roi_values_df.columns = ['%s_%s' % (agg, label.lower()) for label in atlas_loader.labels]

        dfs.append(roi_values_df)

    return pd.concat(dfs, axis=1)


def getAtlasNumVoxels(
        atlas: str,
        atlas_kw: dict = None,
        ref_img: NiftiContainer = None) -> pd.DataFrame:
    """ Returns the number of voxels in each ROI for a given atlas. """
    checkMultiInputTypes(
        ('atlas',     atlas,     [str]),
        ('ref_img',   ref_img,   [NiftiContainer, type(None)]),
        ('atlas_kw',  atlas_kw,  [dict, type(None)]))

    atlas_kw = {} if atlas_kw is None else atlas_kw

    if atlas not in ATLASES:
        raise TypeError(f'Atlas {atlas} not found. Available atlases are {list(ATLASES.keys())}')

    if atlas != 'aal':
        warnings.warn(
            'This function has only been tested for the AAL atlas. Unexpected behaviour may occur for other atlases.')
    atlas_loader = ATLASES[atlas](**atlas_kw)
    atlas_img = NiftiContainer(atlas_loader.maps)

    # if a reference image has been provided resample the atlas to the image dimensions
    if ref_img is not None:
        atlas_img = resampleToImg(atlas_img, ref_img)

    atlas_data = atlas_img.data
    region_id, region_count = np.unique(atlas_data, return_counts=True)
    num_voxels = pd.DataFrame(region_count[1:]).T   # Exclude first region count (not brain regions, checked)
    num_voxels.columns = [label.lower() for label in atlas_loader.labels]

    return num_voxels


def extractROIVoxels(
        images: List[nibabel.nifti1.Nifti1Image or NiftiContainer or CachedNiftiContainer] or nibabel.nifti1.Nifti1Image,
        atlas: str,
        roi: str,
        atlas_kw: dict = None,
        clear_cache: bool = False,
        n_jobs: int = 1,
        verbose: bool = False) -> pd.DataFrame:
    """ Function that allows to extract all the metabolism values of a given ROI belonging to the specified brain
    atlas. """
    def __worker__(_img: nibabel.nifti1.Nifti1Image or CachedNiftiContainer or NiftiContainer, _roi_locator: int,
                   _atlas_nifti: nibabel.nifti1.Nifti1Image, _atlas_data: np.ndarray, _n: int, _clear_cache: bool
                   ) -> tuple:
        if isinstance(_img, nibabel.nifti1.Nifti1Image):   # convert nibable.nifti1.NiftiImage to NiftiContainer
            _img = NiftiContainer(_img)

        # Resample image to atlas
        _res_img = resampleToImg(_img, _atlas_nifti, inplace=False)

        # Check shape
        for _i, (_img_dim, _atlas_dim) in enumerate(zip(_res_img.shape, _atlas_data.shape)):
            assert _img_dim == _atlas_dim, \
                'Image (%d) must be equal shape than atlas (%d) along axis %d' % (_img_dim, _atlas_dim, _i)

        _res_img_data = _res_img.data

        # some images can contain an extra dimension
        if (len(_res_img_data.shape) > len(_atlas_data.shape)) and _res_img_data.shape[-1] == 1:
            _res_img_data = _res_img_data.reshape(_res_img_data.shape[:-1])

        _voxel_values = _res_img_data[np.where(_atlas_data == _roi_locator)]  # select voxels associated with the ROI

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

    atlas_nifti = nifti.loadImg(atlas_loader.maps)
    atlas_data = np.array(atlas_nifti.get_fdata()).astype(int)
    unique_atlas_values = np.unique(atlas_data)
    assert len(unique_atlas_values) > len(atlas_rois), \
        'Unique  (%d) values greater or equal than the number of rois (%d).' \
        % (len(unique_atlas_values), len(atlas_rois))

    # +1 because 0 index correspond to out of the brain voxels
    roi_locator = np.unique(atlas_data)[atlas_rois.index(roi) + 1]

    iterator = tqdm(enumerate(images), desc='Extracting voxels...') if verbose else enumerate(images)
    if n_jobs == 1:
        df = dict([
            __worker__(_img=img, _roi_locator=roi_locator, _atlas_nifti=atlas_nifti, _atlas_data=atlas_data,
                       _n=n, _clear_cache=clear_cache)
            for n, img in iterator])
    else:
        df = dict(
            joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                joblib.delayed(__worker__)(
                    _img=img, _roi_locator=roi_locator, _atlas_nifti=atlas_nifti, _atlas_data=atlas_data,
                    _n=n, _clear_cache=clear_cache) for n, img in iterator))
    df = pd.DataFrame(df).T.sort_index()
    df.columns = ['%s_%d' % (roi, n) for n in range(len(df.columns))]

    return df


def extractMetaROI(
        images: List[nibabel.nifti1.Nifti1Image or NiftiContainer] or nibabel.nifti1.Nifti1Image or NiftiContainer,
        atlas: str,
        rois: list,
        min_voxel_value: float = 0,
        meta_roi_name: str = 'metaROI',
        aggregate: str or list = 'mean',
        atlas_kw: dict = None,
        n_jobs: int = 1,
        resample_to_atlas: bool = True,
        verbose: bool = False) -> pd.DataFrame:
    """ Function that allows to extract several aggregated ROIs provided as a list"""
    def __worker__(_img, _ref_img, _agg, _min_voxel_value, _resample_to_atlas, _n) -> Tuple[int, np.ndarray]:
        if isinstance(_img, NiftiContainer):
            _img = _img.nifti_img
        if isinstance(_ref_img, NiftiContainer):
            _ref_img = _ref_img.nifti_img

        # resample atlas to the image shape
        if _resample_to_atlas:   # resample image to atlas dimensions
            _img = nifti.resampleToImg(_img, _ref_img, interpolation='nearest')
        else:   # resample atlas to image dimensions
            _ref_img = nifti.resampleToImg(_ref_img, _img, interpolation='nearest')

        if _agg is None:  # return all values
            _ref_img_data = np.array(_ref_img.get_fdata())
            _img_data = np.array(_img.get_fdata())
            # check if the reference image is a valid binary mask
            _unique_values = np.unique(_ref_img_data)
            assert len(_unique_values) == 2, 'Wrong mask'
            assert np.sum(_unique_values) == 1, 'Wrong mask'

            if len(_img_data.shape) == 4 and _img_data.shape[-1] == 1:
                _img_data = _img_data.squeeze(-1)  # remove the last dimension
            # select non zero values
            _ref_img_data = _ref_img_data.reshape(-1)
            _img_data     = _img_data.reshape(-1)
            _out_data = _img_data[_ref_img_data > 0]

            # apply minimum voxel value filter
            _out_data = _out_data[_out_data > min_voxel_value]

            return _n, _out_data

        else:  # return statistic
            # create masker
            _masker = NiftiLabelsMasker(
                labels_img=_ref_img, background_label=0,
                standardize=False, strategy='standard_deviation' if _agg == 'std' else _agg
            )
            return _n, _masker.fit_transform(_img).squeeze(0)

    checkMultiInputTypes(
        ('images',            images,             [list, nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('atlas',             atlas,              [str]),
        ('rois',              rois,               [list]),
        ('resample_to_atlas', resample_to_atlas,  [bool]),
        ('aggregate',         aggregate,          [str, list, type(None)]),
        ('atlas_kw',          atlas_kw,           [dict, type(None)]),
        ('n_jobs',            n_jobs,             [int]))

    if atlas not in ATLASES:
        raise TypeError(f'Atlas {atlas} not found. Available atlases are {list(ATLASES.keys())}')

    atlas_kw = {} if atlas_kw is None else atlas_kw
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
    if aggregate is None:
        aggregate = [None]
    else:
        aggregate = [aggregate] if not isinstance(aggregate, list) else aggregate
        for agg in aggregate:
            if agg not in AVAILABLE_AGGREGATIONS:   # None value allowed
                raise TypeError(
                    f'Aggregation strategy {agg} not found. Available strategies are {AVAILABLE_AGGREGATIONS}')
    images = [images] if not isinstance(images, list) else images

    if n_jobs <= 0:
        raise TypeError('n_jobs must be greater than 0 or -1.')

    # load the atlas maps
    atlas_loader = ATLASES[atlas](**atlas_kw)
    atlas_img = NiftiContainer(atlas_loader.maps)

    # convert atlas labels and input rois to lowercase
    atlas_loader.labels = list(map(lambda c: c.lower(), atlas_loader.labels))
    rois = list(map(lambda c: c.lower(), rois))

    # create a meta roi mask with 1s in the input rois
    meta_roi_indices = np.array(
        [int(atlas_loader.indices[atlas_loader.labels.index(roi)]) for roi in rois])
    meta_roi_mask = np.zeros(shape=atlas_img.shape)
    for roi_idx in meta_roi_indices:
        meta_roi_mask += (atlas_img.data == roi_idx).astype(int)
    meta_roi_mask[meta_roi_mask > 0] = 1

    # create the masked atlas
    masked_atlas_img = NiftiContainer.fromData(data=meta_roi_mask, ref_image=atlas_img.nifti_img)

    # extract meta-roi values
    dfs = []
    for agg in aggregate:
        iterator = tqdm(enumerate(images), desc='Extracting ROIs...') if verbose else enumerate(images)
        if n_jobs == 1:
            roi_values = dict(__worker__(img, masked_atlas_img, agg, min_voxel_value, resample_to_atlas, n)
                              for n, img in iterator)
        else:
            roi_values = dict(
                joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                    joblib.delayed(__worker__)(img, masked_atlas_img, agg, min_voxel_value, resample_to_atlas, n)
                    for n, img in iterator))

        roi_values_df = pd.DataFrame(roi_values).T.sort_index()
        # Add prefix when several aggregations are provided
        if agg is None:
            roi_values_df.columns = ['%s_v%d' % (meta_roi_name, i) for i in range(roi_values_df.shape[1])]
        elif len(aggregate) == 1:
            roi_values_df.columns = [meta_roi_name]
        else:
            roi_values_df.columns = ['%s_%s' % (agg, meta_roi_name)]

        dfs.append(roi_values_df)

    return pd.concat(dfs, axis=1)
