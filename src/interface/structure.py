from __future__ import division
import numpy as np
class Structure(object):
    """
    A class to represent the data.
    """
    def __init__(self):
        self.parameters = dict()
        self.geometry = []
        self.maximum_eps = 0
        self.maximum_mu = 0

    def add_parameter(self, parameter_name, value):
        self.parameters[parameter_name] = value;

    def add_geometry(self, geometry_object):
        self.geometry.append(geometry_object)

    def create_dimensions(self):
        x_index_dim = self.parameters.get("x_index_dim", None)
        y_index_dim = self.parameters.get("y_index_dim", None)
        if not x_index_dim or not y_index_dim:
            xdim = self.parameters.get("xdim", None)
            ydim = self.parameters.get("ydim", None)
            delta = self.parameters.get("delta", None)
            if any((xdim, ydim, delta)) is None:
                raise ValueError("Not enough data provided to calculate"
                                 " dimensions of the grid")
            x_index_dim = xdim / delta;
            y_index_dim = ydim / delta;
        self.x_index_dim = x_index_dim

    def create_arrays(self):
        simulation_type = self.parameters.get("simulation_type", "TM_SIMULATION")
        if simulation_type == "TM_SIMULATION":
            self.epsilon_array = np.zeros((self.x_index_dim, self.y_index_dim),
                                          dtype = 'float64')
            self.mu_array = np.zeros((self.x_index_dim, self.y_index_dim),
                                     dtype = 'float64')
            self.sigma_array = np.zeros((self.x_index_dim, self.y_index_dim),
                                        dtype = 'float64')
            self.sigma_star_array = np.zeros((self.x_index_dim, self.y_index_dim),
                                             dtype = 'float64')
