import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class PointsEditor(object):
    
    title = ("Click and drag a marker to move it; "
            "'a' to add; 'd' to delete; 'v' set visibility.\n"
            "Close figure when done.")
    
    def __init__(self, ax, points=None, max_distance=30):
        
        self._ax = ax
        
        if points is None:
            self._points = np.zeros((0,2))
        else:
            self._points = points
        
        self.max_distance = max_distance
        
        self.visible = True
    
    @property
    def points(self):
        return self._points
    
    def edit(self, close_callback=None, **kwargs):
        
        self._ind = None
        self.create_artists(self._ax, **kwargs)
        
        canvas = self._line.figure.canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        
        if close_callback is not None:
            self.close_callback_fun = close_callback
            canvas.mpl_connect('close_event', self.close_callback)
        
        self.canvas = canvas
    
    def close_callback(self, event):
        self.close_callback_fun(self._points)
    
    def create_artists(self, ax, **kwargs):
        x,y = self._points.T
        
        self._line = ax.scatter(x, y, animated=True, **kwargs)
        
        ax.set_clip_on(False)
        ax.set_title(self.title)

    def draw_artists(self):
        self._ax.draw_artist(self._line)
    
    def update_artists(self):
        self.canvas.restore_region(self._background)
        self._line.set_offsets(self._points)
        self.draw_artists()
        self.canvas.update()
        self.canvas.flush_events()
        
    def set_visible(self):
        self.visible = not self.visible
        self._line.set_visible(self.visible)
        if not self.visible:
            self._ind = None
        
    def draw_callback(self, event):
        self._background = self.canvas.copy_from_bbox(self._ax.bbox)
        self.draw_artists()
        self.canvas.update()
    
    def button_press_callback(self, event):
        ignore = (not self.visible or event.inaxes is None or event.button != 1 or 
                    len(self._points) == 0)
        if ignore:
            return
        self._ind = self.get_ind_under_cursor(event)

    def button_release_callback(self, event):
        ignore = not self.visible or event.button != 1
        if ignore:
            return
        self._ind = None
    
    def key_press_callback(self, event):
        if not event.inaxes:
            return
        if event.key=='v':
            self.set_visible()
        elif event.key=='d':
            self.delete_marker(event)
        elif event.key=='a':
            self.add_marker(event)
        
        self.canvas.update()

    def motion_notify_callback(self, event):
        ignore = (not self.visible or event.inaxes is None or
                  event.button != 1 or self._ind is None or 
                  len(self._points) == 0)
        
        if ignore:
            return
        x,y = event.xdata, event.ydata
        
        self._points[self._ind][0] = x
        self._points[self._ind][1] = y
        
        self.update_artists()
        
    def delete_marker(self,event):
        if len(self._points) == 0:
            return
        ind = self.get_ind_under_cursor(event)
        if ind is None:
            return
        self._points = np.delete(self._points, ind, axis=0)
        
        self.update_artists()
        
    def add_marker(self,event):
        x,y = event.xdata, event.ydata
        self._points = np.vstack((self._points, np.array([[x,y]])))
        self.update_artists()
    
    def get_ind_under_cursor(self, event):
        
        dist = cdist([[event.xdata, event.ydata]], self._points)
        closest_ind = np.argmin(dist)
        
        if dist[0][closest_ind] >= self.max_distance:
            closest_ind = None
        
        return closest_ind