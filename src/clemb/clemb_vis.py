"""
Code to visualize results from the crater lake energy and mass balance 
computations.
"""

from matplotlib.lines import Line2D
import numpy as np


class LineInteractor(object):
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, line):
        self.ax = ax
        canvas = line.figure.canvas

        x, y = line.get_data()
        self.xd = x.copy()
        self.yd = y.copy()
        self.line = Line2D(x, y, marker='o', markerfacecolor='r')
        self.ax.add_line(self.line)

        self._ind = None  # the active vert

        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect(
            'button_release_event', self.button_release_callback)
        #canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        x, y = self.line.get_data()
        xyt = self.line.get_transform().transform(np.vstack((x, y)).T)
        xt = xyt[:, 0]
        yt = xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        xt, yt = self.line.get_data()
        yt[self._ind] = y
        self.line.set_data(xt, yt)
        self.line.figure.canvas.draw()
        self._ind = None

    def motion_notify_callback(self, event):
        'on mouse movement'
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        xt, yt = self.line.get_data()
        yt[self._ind] = y
        self.line.set_data(xt, yt)
        self.line.figure.canvas.draw()

    def reset(self):
        x = self.xd.copy()
        y = self.yd.copy()
        self.line.set_data(x, y)
        self.line.figure.canvas.draw()

    def get_data(self):
        return self.line.get_data()
