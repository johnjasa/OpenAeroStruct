import unittest
import numpy as np
from openmdao.api import Group, IndepVarComp

from openaerostruct.aerodynamics.eval_mtx import EvalVelMtx
from openaerostruct.utils.testing import run_test, get_default_surfaces


np.random.seed(14)

class Test(unittest.TestCase):

    def test(self):
        surfaces = get_default_surfaces()
        num_eval_points = 2
        nx = surfaces[0]['num_x']
        ny = surfaces[0]['num_y']

        group = Group()

        ivp = IndepVarComp()
        ivp.add_output('wing_test_name_vectors', np.random.random_sample((num_eval_points, nx, 2*ny-1, 3)))

        group.add_subsystem('ivp', ivp, promotes=['*'])
        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=num_eval_points, eval_name='test_name')

        group.add_subsystem('comp', comp, promotes=['*'])

        run_test(self, group, compact_print=False)


if __name__ == '__main__':
    unittest.main()
