# Module with tools for image renderization
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import matplotlib.pyplot as plt
import nibabel
import warnings
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter("always")

    from nilearn import datasets as nl_data
    from nilearn import plotting as nl_plotting
    from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
    from nilearn import image as nl_image

from .image import (
    NiftiContainer,
    resampleVoxelSize)
from .decorator import ignore_warning
from .validation import (
    checkInputType,
    checkMultiInputTypes)


@ignore_warning
def render(img: nibabel.nifti1.Nifti1Image or NiftiContainer, backend: str = 'dynamic', title: str = '',
           vmax: float or int = None, vmin: float or int = None, cmap: str = 'cold_hot', colorbar: bool = True,
           cut_coords: tuple or float or int = None, annotate: bool = False, threshold: float = None,
           figsize: tuple = (20, 6), save_html: str = None, save_png: str = None, open_in_browser: bool = False, **_):
    """ Function to render nifti images. This function can render images statically (fixed image that cannot be
    interacted with) or dynamically (image that can be interacted with) by selecting the backend as "static" or
    "dynamic" respectively.

    :param nibabel.nifti1.Nifti1Image or NiftiContainer img: image to be rendered.
    :param str backend: backend used to render the image. Available backends are: "static" and "dynamic". Defaults
        to "dynamic".
    :param str title: title. Defaults to "".
    :param float or int vmax: upper bound of the colormap. If None, the max of the image is used. Defaults to None.
    :param float or int vmin: lower bound of the colormap. If None, the min of the image is used. Only available for
        "dynamic" backend. Defaults to None.
    :param str cmap: colormap. Defaults to "cold_hot".
    :param bool colorbar: if True, display a colorbar in the plots. Defaults to True.
    :param tuple or float or int cut_coords: the MNI coordinates of the point where the cut is performed as a 3-tuple:
        (x, y, z). If None is given, the cuts are calculated automatically. Defaults to None.
    :param bool annotate: if annotate is True, current cuts are added to the viewer. Default to False.
    :param float threshold: if None is given, the image is not thresholded. If a string of the form “90%” is given,
        use the 90-th percentile of the absolute value in the image. If a number is given, it is used to threshold the
        image: values below the threshold (in absolute value) are plotted as transparent. If auto is given, the
        threshold is determined automatically. Only available for "dynamic" backend. Defaults to 1e-06.
    :param tuple figsize: figure size. Only available for "static" backend. Defaults to (20, 6).
    :param str save_png: save the image as a png image. Only available for "static" backend. When this parameter is set
        to a value the image will not be displayed. Defaults to None.
    :param str save_html: save the visualization as html. Only available for "dynamic" backend. Defaults to None.
    :param bool open_in_browser: Open the visualization in a browser. Only available for "dynamic" backend.
        Defaults to False.
    """
    checkMultiInputTypes(
        ('img',             img,             [nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('backend',         backend,         [str]),
        ('title',           title,           [str]),
        ('vmax',            vmax,            [float, int, type(None)]),
        ('vmin',            vmin,            [float, int, type(None)]),
        ('cmap',            cmap,            [str]),
        ('colorbar',        colorbar,        [bool]),
        ('cut_coords',      cut_coords,      [float, int, tuple, type(None)]),
        ('annotate',        annotate,        [bool]),
        ('threshold',       threshold,       [float, type(None)]),
        ('figsize',         figsize,         [tuple, list]),
        ('save_html',       save_html,       [str, type(None)]),
        ('open_in_browser', open_in_browser, [bool]))

    if backend not in ['static', 'dynamic']:
        raise TypeError(f'Invalid backend "{backend}". Available backends are "static" and "dynamic".')

    if isinstance(img, NiftiContainer):
        img = img.nifti_img

    if backend == 'static':
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('black')
        nl_plotting.plot_stat_map(
            img, bg_img=img, axes=ax, title=title, vmax=vmax, cmap=nilearn_cmaps[cmap], colorbar=colorbar,
            cut_coords=cut_coords, annotate=annotate)

        if save_png is not None:
            plt.savefig(save_png, dpi=300, format='png')
        else:
            plt.show()
    elif backend == 'dynamic':
        html_view = nl_plotting.view_img(
            img, bg_img=img, cut_coords=cut_coords, title=title, annotate=annotate, vmax=vmax,
            threshold=threshold, vmin=vmin, symmetric_cmap=True if vmax == vmin else False)

        if save_html is not None:
            checkInputType('save_html', save_html, [str])
            html_view.save_as_html(save_html)
        if open_in_browser:
            html_view.open_in_browser()
        return html_view
    else:
        assert False, 'I should not appear.'


def render3D(img: nibabel.nifti1.Nifti1Image or NiftiContainer, rescale: bool = True, isomin: float = 0.05,
             isomax: float = 1.0, opacity: float = 0.3, resolution: int = 10, cmap: str = 'Turbo',
             template: str = 'plotly_white', title: str = '', width: int = 1000, height: int = 800,
             resample_voxel_size: int or tuple or float = 5, open_in_browser: bool = False, show: bool = False,
             **_):
    """ Function that renders a nifti image in 3 dimensions using plotly as backend.

    :param nibabel.nifti1.Nifti1Image or NiftiContainer img: image to be rendered.
    :param bool rescale: rescale the image values to the range [0-1]. Defaults to True.
    :param float isomin: lower value of the representation. Defaults to 0.05.
    :param float isomax: upper value of the representation. Defaults to 1.0.
    :param float opacity: representation opacity. Defaults to 0.3.
    :param int resolution: image resolution. It is recommended to keep this value lower to avoid overload the browser.
        Defaults to 10.
    :param str cmap: colormap. Defaults to "Turbo".
    :param str template: plotly backend template. Defaults to "plotly_white".
    :param str title: title. Defaults to "".
    :param int width: image width. Defaults to 1000.
    :param int height: image height. Defaults to 800.
    :param int or float or tuple resample_voxel_size: voxel size to which the image will be resampled. It is
        recommended to keep this value higher to avoid overload the browser. Defaults to 5.
    :param bool open_in_browser: Open the visualization in a browser. Defaults to False.
    :param bool show: indicates whether to display the figure or not. Defaults to True.

    IMPORTANT NOTE: It is recommended to leave the default parameters resample_voxel_size and resolution to reduce the
    display load and avoid breaking the browser.
    """
    checkMultiInputTypes(
        ('img',                 img,                 [nibabel.nifti1.Nifti1Image, NiftiContainer]),
        ('rescale',             rescale,             [bool]),
        ('isomin',              isomin,              [float]),
        ('isomax',              isomax,              [float]),
        ('opacity',             opacity,             [float]),
        ('resolution',          resolution,          [int]),
        ('cmap',                cmap,                [str]),
        ('template',            template,            [str]),
        ('title',               title,               [str]),
        ('width',               width,               [int]),
        ('height',              height,              [int]),
        ('resample_voxel_size', resample_voxel_size, [int, tuple, float]))

    if resample_voxel_size is not None:
        img = resampleVoxelSize(img, voxel_size=resample_voxel_size, inplace=False)

    data = img.data

    # Rescale values to the range 0-1
    if rescale:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    volumen = np.transpose(data, (1, 0, 2))  # rearange dimensions
    X, Y, Z = np.mgrid[0:volumen.shape[0], 0:volumen.shape[1], 0:volumen.shape[2]]

    # Render image
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volumen.flatten(),
        isomin=isomin,
        isomax=isomax,
        opacity=opacity,  # needs to be small to see through all surfaces
        surface_count=resolution,  # needs to be a large number for good volume rendering
        colorscale=cmap))

    # Layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        template=template)

    if open_in_browser:
        app = dash.Dash()
        app.layout = html.Div([dcc.Graph(figure=fig)])
        app.run_server(debug=True, use_reloader=False)
    elif show:
        fig.show()

    return fig


def renderAAL(regions: list, weights: list or np.ndarray = None, default_val: float = 0.0, backend: str = 'render',
              backend_kw: dict = None):
    """ Function to render regions of the AAL (Automated Anatomical Labelling) atlas.

    :param list regions: list of regions belonging to the AAL atlas to be rendered.
    :param list or np.array weights: list of weights associated with each of the regions in "regions" to be assigned
        to each region. Defaults to None.
    :param float default_val: value to assign to regions that are not in the input regions. Defaults to 0.0.
    :param str backend: backend used to render the image. Available backends are "render" (uses mitools.plot.render
        function) and "render3D" (uses mitools.plot.render3D). Defaults to "render".
    :param dict backend_kw: optional arguments used for the corresponding backend function. See documentation
        associated to mitools.plot.render and mitools.plot.render3D for "render" and "render3D" backends respectively.
    """
    checkMultiInputTypes(
        ('regions',     regions,     [list]),
        ('weights',     weights,     [list, np.ndarray, type(None)]),
        ('default_val', default_val, [float]),
        ('backend',     backend,     [str]),
        ('backend_kw',  backend_kw,  [dict, type(None)]))

    backend_kw = {} if backend_kw is None else backend_kw

    if backend not in _AVAILABLE_RENDER_BACKENDS:
        raise TypeError(
            f'Invalid "backend" {backend}. Available backends are: {list(_AVAILABLE_RENDER_BACKENDS.keys())}.')

    if weights is not None:
        if not len(weights) == len(regions):
            raise TypeError(f'"regions" ({len(regions)}) and "weights" ({len(weights)}) must have the same length.')
        region_weight = dict(zip(regions, list(weights)))
    else:
        region_weight = {region: 1.0 for region in regions}

    aal_atlas = nl_data.fetch_atlas_aal()
    aal = nl_image.load_img(aal_atlas.maps)
    aal_data = np.array(aal.get_fdata())
    aal_labels = [label.lower() for label in aal_atlas['labels']]

    # Check and modify input regions
    region_weight_ = {}
    for region in region_weight.keys():
        if region.lower() not in aal_labels:
            raise TypeError(f'Region "{region}" not in AAL labels ({aal_labels}).')
        region_weight_[region.lower()] = region_weight[region]
    region_weight = region_weight_

    # Get the coordinates associated with each region
    aal_coords = [float(v) for v in aal_atlas['indices']]
    region_id = dict(zip(aal_labels, aal_coords))
    region_coords = {region: np.where(aal_data == id) for region, id in region_id.items()}

    # Create template to plot
    aal_template = np.zeros(shape=aal_data.shape)
    for region, coords in region_coords.items():
        if region in regions:
            aal_template[coords] = region_weight[region]
        else:
            aal_template[coords] = default_val  # non-target regions will be plotted with the default val

    nii_obj = NiftiContainer.fromData(data=aal_template, ref_image=aal_atlas.maps)

    return _AVAILABLE_RENDER_BACKENDS[backend](img=nii_obj, **backend_kw)


# Available rendering functions
_AVAILABLE_RENDER_BACKENDS = {
    'render': render,
    'render3D': render3D
}

