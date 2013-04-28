from __future__ import division
import numpy as np

class Geometry(object):
    def apply(self, array, greater = True, use_original_dimensions = False):
        return NotImplementedError("The apply method has not been implemented")


class Line(Geometry):
    """
    The line is of the form y = ax + by = c
    You need to provide two parameters namely m and c
    """
    def __init__(self, value, array_type, greater, a, b, c):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.value = float(value)
        self.array_type = array_type
        if greater == 'GREATER':
            self.greater = True
        else:
            self.greater = False

    def apply(self, array):
        range_x = np.arange(array.shape[1])
        range_y = np.arange(array.shape[0])
        X, Y = np.meshgrid(range_x, range_y)
        if self.greater:
            array[np.where( self.a * X + self.b * Y > self.c)] = self.value
        else:
            array[np.where(self.a * X + self.b * Y < self.c)] = self.value

    def __repr__(self):
        return " Line with a = %d, b=%d, c = %d" % (self.a, self.b, self.c)


class Ellipse(Geometry):
    """
    Implements the geometry of an Ellipse
    """
    def __init__(self, value, array_type, greater,  centre_x, centre_y, a, b):
        self.array_type = array_type
        self.centre_x = float(centre_x)
        self.centre_y = float(centre_y)
        if greater == 'GREATER':
            self.greater = True
        else:
            self.greater = False
        self.a = float(a)
        self.b = float(b)
        self.value = float(value)

    def apply(self, array):
        range_x = np.arange(array.shape[1])
        range_y = np.arange(array.shape[0])
        X, Y = np.meshgrid(range_x, range_y)
        if self.greater:
            ii = np.where((X - self.centre_x) ** 2 / self.a**2 + (Y - self.centre_y) ** 2 / self.b**2 > 1)
            array[ii] = self.value
        else:
            ii = np.where((X - self.centre_x) ** 2 / self.a**2 + (Y - self.centre_y) ** 2 / self.b**2 < 1)
            array[ii] = self.value


class Circle(Ellipse):
    """
    Implements the geometry of a Circle

    """
    def __init__(self, value, array_type, greater, centre_x, centre_y, radius):
        super(Circle, self).__init__(value, array_type, greater,  centre_x, centre_y, radius, radius)
