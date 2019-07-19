from datetime import datetime
import unittest

import numpy as np

from clemb.forward_model import Forwardmodel



class ForwardModelTestCase(unittest.TestCase):

    def test_without_q_gradient(self):
        """
        Test forward model assuming no gradient
        of the heat input rate between time steps.
        """
        # First test with Euler
        T0 = 30.
        dt = 1.
        a = 194162
        da = 0.
        v = 8875
        dv = 0.
        Qi0 = 400. * 0.0864
        dQi = 0.
        Mi = 10
        dMi = 0.
        Mo = 5.
        dMo = 0.
        H = 2.8
        dH = 0.
        ws = 4.5
        dws = 0.
        density = 1.003 - 0.00033 * T0
        M0 = v*density
        X0 = 2.
        dtime = datetime(2019, 1, 1, 0, 0, 0)
        npdp = np.dtype({'names': ['T', 'M', 'X', 'Qi', 'Mi', 'Mo', 'a',
                                   'v', 'H', 'ws'],
                         'formats': [float] * 10})
        y0 = [T0, M0, X0, Qi0, Mi, Mo, a, v, H, ws]
        dp = [dQi, dMi, dMo, da, dv, dH, dws]
        fw = Forwardmodel()
        y_new = fw.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.61, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.96, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw.get_evap()[0], 14.48, places=2)
        self.assertAlmostEqual(fw.get_evap()[1], 3.15, places=2)

        # Then test with second-order Runge-Kutta
        fw1 = Forwardmodel(method='rk2')
        y_new = fw1.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.6, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.89, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 14.73, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.22, places=2)

        # Finally test with fourth-order Runge-Kutta
        fw2 = Forwardmodel(method='rk4')
        y_new = fw2.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.56, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.89, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw2.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw2.get_evap()[0], 14.93, places=2)
        self.assertAlmostEqual(fw2.get_evap()[1], 3.27, places=2)

    def test_with_q_gradient(self):
        """
        Test forward model assuming no gradient
        of the heat input rate between time steps.
        """
        # First test with Euler
        T0 = 30.
        dt = 1.
        a = 194162
        da = 0.
        v = 8875
        dv = 0.
        Qi0 = 400. * 0.0864
        dQi = (1000. - 400.) * 0.0864
        Mi = 10
        dMi = 0.
        Mo = 5.
        dMo = 0.
        H = 2.8
        dH = 0.
        ws = 4.5
        dws = 0.
        density = 1.003 - 0.00033 * T0
        M0 = v*density
        X0 = 2.
        dtime = datetime(2019, 1, 1, 0, 0, 0)
        fw = Forwardmodel()
        npdp = np.dtype({'names': ['T', 'M', 'X', 'Qi', 'Mi', 'Mo', 'a',
                                   'v', 'H', 'ws'],
                         'formats': [float] * 10})
        y0 = [T0, M0, X0, Qi0, Mi, Mo, a, v, H, ws]
        dp = [dQi, dMi, dMo, da, dv, dH, dws]
        fw = Forwardmodel()
        y_new = fw.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.61, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.96, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw.get_evap()[0], 14.48, places=2)
        self.assertAlmostEqual(fw.get_evap()[1], 3.15, places=2)

        # Then test with second-order Runge-Kutta
        fw1 = Forwardmodel(method='rk2')
        y_new = fw1.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 31.3, places=2)
        self.assertAlmostEqual(y_new['M'], 8837.15, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 21.6, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 14.73, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.22, places=2)

        # Finally test with fourth-order Runge-Kutta
        fw2 = Forwardmodel(method='rk4')
        y_new = fw2.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 31.26, places=2)
        self.assertAlmostEqual(y_new['M'], 8837.1, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw2.get_steam(), 30.86, places=2)
        self.assertAlmostEqual(fw2.get_evap()[0], 15.53, places=2)
        self.assertAlmostEqual(fw2.get_evap()[1], 3.44, places=2)


if __name__ == '__main__':
    unittest.main()
