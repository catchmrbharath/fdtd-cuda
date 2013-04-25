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
        self.x_dim = int(self.parameters.get("x_dim", None))
        self.y_dim= int(self.parameters.get("y_dim", None))
        if any((self.x_dim, self.y_dim)) is None:
            raise Exception("The dimensions could not be read"
                            " from the file")

    def create_arrays(self):
        simulation_type = self.parameters.get("simulation_type", "TM_SIMULATION")
        if simulation_type == "TM_SIMULATION":
            self.epsilon_array = np.zeros((self.x_dim, self.y_dim),
                                          dtype = 'float64')
            self.mu_array = np.zeros((self.x_dim, self.y_dim),
                                     dtype = 'float64')
            self.sigma_array = np.zeros((self.x_dim, self.y_dim),
                                        dtype = 'float64')
            self.sigma_star_array = np.zeros((self.x_dim, self.y_dim),
                                             dtype = 'float64')
