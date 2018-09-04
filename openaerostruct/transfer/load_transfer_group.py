from openmdao.api import Group, ExplicitComponent, BsplinesComp

from openaerostruct.transfer.compute_aerostruct_points import ComputeAerostructPoints
from openaerostruct.transfer.load_transfer import LoadTransfer



from openmdao.api import IndepVarComp, Group


class LoadTransferGroup(Group):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        # Add structural components to the surface-specific group
        self.add_subsystem('compute_aerostruct_points',
                 ComputeAerostructPoints(surface=surface),
                 promotes=['*'])

        # Add structural components to the surface-specific group
        self.add_subsystem('load_transfer',
                 LoadTransfer(surface=surface),
                 promotes=['*'])
