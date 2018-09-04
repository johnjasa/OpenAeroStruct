from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent


class ComputeAerostructPoints(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = surface['num_y']
        self.nx = surface['num_x']

        if surface['fem_model_type'] == 'tube':
            self.fem_origin = surface['fem_origin']
        else:
            y_upper = surface['data_y_upper']
            x_upper = surface['data_x_upper']
            y_lower = surface['data_y_lower']

            self.fem_origin = (x_upper[0]  * (y_upper[0]  - y_lower[0]) +
                               x_upper[-1] * (y_upper[-1] - y_lower[-1])) / \
                             ((y_upper[0]  -  y_lower[0]) + (y_upper[-1] - y_lower[-1]))

        self.add_input('def_mesh', val=np.ones((self.nx, self.ny, 3)), units='m')

        self.add_output('aero_points', val=np.ones((self.nx-1, self.ny-1, 3)), units='m')
        self.add_output('struct_points', val=np.ones((self.ny-1, 3)), units='m')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        mesh = inputs['def_mesh']

        # Compute the aerodynamic centers at the quarter-chord point of each panel
        w = 0.25
        outputs['aero_points'] = 0.5 * (1-w) * mesh[:-1, :-1, :] + \
                                 0.5 *   w   * mesh[1:, :-1, :] + \
                                 0.5 * (1-w) * mesh[:-1,  1:, :] + \
                                 0.5 *   w   * mesh[1:,  1:, :]

        # Compute the structural midpoints based on the fem_origin location
        w = self.fem_origin
        outputs['struct_points'] = 0.5 * (1-w) * mesh[0, :-1, :] + \
                                   0.5 *   w   * mesh[-1, :-1, :] + \
                                   0.5 * (1-w) * mesh[0,  1:, :] + \
                                   0.5 *   w   * mesh[-1,  1:, :]
