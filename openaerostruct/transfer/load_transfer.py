from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


class LoadTransfer(ExplicitComponent):
    """
    Perform aerodynamic load transfer.

    Apply the computed sectional forces on the aerodynamic surfaces to
    obtain the deformed mesh FEM loads.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).

    Returns
    -------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = surface['num_y']
        self.nx = surface['num_x']

        self.add_input('aero_points', val=np.ones((self.nx-1, self.ny-1, 3)), units='m')
        self.add_input('struct_points', val=np.ones((self.ny-1, 3)), units='m')
        self.add_input('sec_forces', val=np.ones((self.nx-1, self.ny-1, 3)), units='N')

        # Well, technically the units of this load array are mixed.
        # The first 3 indices are N and the last 3 are N*m.
        self.add_output('loads', val=np.zeros((self.ny, 6)), units='N')

        self.declare_partials('*', '*')

        if not fortran_flag:
            self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        sec_forces = inputs['sec_forces'].copy()
        nx = self.nx
        ny = self.ny

        # Find the moment arm between the aerodynamic centers of each panel
        # and the FEM elements
        diff = inputs['aero_points'] - inputs['struct_points']
        moment = np.zeros((self.ny - 1, 3))
        for ind in range(self.nx-1):
            moment = moment + np.cross(diff[ind, :, :], sec_forces[ind, :, :], axis=1)

        # Compute the loads based on the xyz forces and the computed moments
        loads = outputs['loads']
        loads[:] = 0.
        sec_forces_sum = np.sum(sec_forces, axis=0)
        loads[:-1, :3] = loads[:-1, :3] + 0.5 * sec_forces_sum[:, :]
        loads[ 1:, :3] = loads[ 1:, :3] + 0.5 * sec_forces_sum[:, :]
        loads[:-1, 3:] = loads[:-1, 3:] + 0.5 * moment
        loads[ 1:, 3:] = loads[ 1:, 3:] + 0.5 * moment

        outputs['loads'] = loads

        # ============================

        outputs['loads'] = 0.

        aero_pts = inputs['aero_points']
        struct_pts = np.einsum('j,ikl->ijkl',
            np.ones(num_points_x - 1), inputs[fea_mesh_name][:, :-1, :])
        moments = compute_cross(aero_pts - struct_pts, inputs[forces_name])
        forces = inputs[forces_name]
        outputs[loads_name][:, :-1, :3] += 0.5 * np.sum( forces, axis=1)
        outputs[loads_name][:, :-1, 3:] += 0.5 * np.sum(moments, axis=1)

        aero_pts = inputs[vlm_mesh_name]
        struct_pts = np.einsum('j,ikl->ijkl',
            np.ones(num_points_x - 1), inputs[fea_mesh_name][:, 1:, :])
        moments = compute_cross(aero_pts - struct_pts, inputs[forces_name])
        forces = inputs[forces_name]
        outputs[loads_name][:, 1: , :3] += 0.5 * np.sum( forces, axis=1)
        outputs[loads_name][:, 1: , 3:] += 0.5 * np.sum(moments, axis=1)
