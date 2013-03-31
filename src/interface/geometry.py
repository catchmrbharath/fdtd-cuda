from __future__ import division
import numpy as np

class Geometry(object):
    def __init__(self, dx):
        self.dx = dx
    def apply(self, array, greater = True, use_original_dimensions = False):
        return NotImplementedError("The apply method has not been implemented")


class Line(Geometry):
    """
    The line is of the form y = m * x + c
    You need to provide two parameters namely m and c
    """
    def __init__(self, dx, m, c):
        super(Geometry, self).__init__(dx)
        self.m = m
        self.c = c
    def apply(self, array, value, greater = True,
              use_original_dimensions=False):
        if use_original_dimensions:
            self.c = self.c / self.dx

        range_x = np.arange(array.shape[1])
        range_y = np.arange(array.shape[0])
        X, Y = np.meshgrid(range_x, range_y)
        if greater:
            array[np.where( Y > self.m * X + self.c)] = value
        else:
            array[np.where( Y < self.m * X + self.c)] = value


class Ellipse(Geometry):
    """
    Implements the geometry of an Ellipse
    """
    def __init__(self, dx, a, b):
        super(Geometry, self).__init__(dx)
        self.a = a
        self.b = b

    def apply(self, array, value, greater = True,
              use_original_dimensions=False):
        if use_original_dimensions:
            self.a = self.a / self.dx
            self.b = self.b / self.dx
        range_x = np.arange(array.shape[1])
        range_y = np.arange(array.shape[0])
        X, Y = np.meshgrid(range_x, range_y)
        if greater:
            ii = np.where(X * X / self.a**2 + Y * Y / self.b**2 > 1)
            array[ii] = value
        else:
            ii = np.where(X * X / self.a**2 + Y * Y / self.b**2 < 1)
            array[ii] = value


class Circle(Ellipse):
    """
    Implements the geometry of a Circle

    """
    def __init__(self, dx, radius):
        super(Ellipse, self).__init__(self, dx, radius, radius)
