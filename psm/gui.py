from collections import OrderedDict

import numpy as np
import bqplot
from bqplot.interacts import BrushSelector, PanZoom
import ipywidgets as widgets
from traitlets import link
import PIL
import io


def axis2name(axis):
    if axis is 0:
        return 'x'
    elif axis is 1:
        return 'y'


def array2widget(array):
    array = (array - array.min()) / (array.max() - array.min()) * 255

    pil_image = PIL.Image.fromarray(array).convert('RGB')
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='png')
    return widgets.Image(value=byte_arr.getvalue(), format='png')


class Element(object):

    def __init__(self, figure, scatter):
        self._scatter = scatter
        self._figure = figure

    figure = property(lambda self: self._figure)
    scatter = property(lambda self: self._scatter)
    axes = property(lambda self: self.figure.axes)
    scales = property(lambda self: [self.axes[0].scale, self.axes[1].scale])


class Navigator(Element):

    def __init__(self, figure, scatter, default_region):
        super().__init__(figure, scatter)
        self._default_region = np.array(default_region).astype(np.float)

        pan_zoom = PanZoom(scales={'x': [self.scales[0]], 'y': [self.scales[1]]})
        brush = BrushSelector(x_scale=self.scales[0], y_scale=self.scales[1], marks=[scatter])
        toggle_interact = widgets.ToggleButtons(options=OrderedDict([('Pan & Zoom', pan_zoom),
                                                                     ('Select region', brush)]))
        link((toggle_interact, 'value'), (self.figure, 'interaction'))

        set_region_button = widgets.Button(description='Set region', disabled=False, button_style='')

        def on_button_clicked(_):
            if brush.selected is not None:
                self.set_region(brush.selected)
                brush.selected = None

        set_region_button.on_click(on_button_clicked)

        reset_region_button = widgets.Button(description='Reset region', disabled=False, button_style='')
        reset_region_button.on_click(lambda _: self.set_region())

        button_box = widgets.HBox([toggle_interact, set_region_button, reset_region_button])
        region_box = widgets.HBox([self._build_region_input(0, 'min'), self._build_region_input(0, 'max'),
                                   self._build_region_input(1, 'min'), self._build_region_input(1, 'max')])

        self._accordion = widgets.Accordion(children=[widgets.VBox([button_box, region_box])])
        self._accordion.set_title(0, 'Navigation')

    def set_region(self, limits=None):
        if limits is None:
            limits = self._default_region

        self.scales[0].min = limits[0, 0]
        self.scales[0].max = limits[1, 0]
        self.scales[1].min = limits[0, 1]
        self.scales[1].max = limits[1, 1]

    def _build_region_input(self, axis, min_or_max):
        style = {'description_width': 'initial'}

        if min_or_max is 'min':
            default = self._default_region[0][axis]
        elif min_or_max is 'max':
            default = self._default_region[1][axis]
        else:
            raise RuntimeError()

        float_text = widgets.FloatText(value=default, description='{} {}:'.format(axis2name(axis), min_or_max),
                                       disabled=False,
                                       style=style)
        link((float_text, 'value'), (self.scales[axis], min_or_max))
        return float_text


class PointEditor(Element):

    def __init__(self, x, y, background):
        x_scale = bqplot.LinearScale()
        y_scale = bqplot.LinearScale()

        x_axis = bqplot.Axis(scale=x_scale)
        y_axis = bqplot.Axis(scale=y_scale, orientation='vertical')

        scatter = bqplot.ScatterGL(x=x, y=y, scales={'x': x_scale, 'y': y_scale})
        ipyimage = widgets.Image(image=array2widget(background))
        image = bqplot.Image(image=ipyimage, scales={'x': x_scale, 'y': y_scale})

        figure = bqplot.Figure(axes=[x_axis, y_axis], marks=[image])

        super().__init__(figure, scatter)

        default_region = np.array([[x.min(), y.min()], [x.max(), y.max()]])

        self._navigator = Navigator(figure, scatter, default_region)

    def display(self):
        return self.figure
    #    return widgets.VBox([self.figure, self._navigator._accordion])
