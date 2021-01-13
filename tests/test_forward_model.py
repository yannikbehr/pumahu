from datetime import datetime
import unittest

import numpy as np

from .forward_model import Forwardmodel
from .syn_model import SynModel


class ForwardModelTestCase(unittest.TestCase):

    def test_esol(self):
        fm = Forwardmodel()
        e = fm.esol(1, 2e5, datetime(2003,1,19,0,0,0))
        self.assertEqual(e, 4.5)

    def test_surface_loss(self):
        fm = Forwardmodel()
        loss, ev = fm.surface_loss(35.0, 5.0, 200000)
        self.assertAlmostEqual(loss, 20.61027, 5)
        self.assertAlmostEqual(ev, 4.87546, 5)
    
    def test_fullness(self):
        fm = Forwardmodel()
        a, vol = fm.fullness(np.ones(10) * 2529.4)
        fvol = np.ones(10) * 8880.29883
        diffvol = abs(vol - fvol) / fvol * 100.
        fa = np.ones(10) * 196370.188
        diffa = abs(a - fa) / fa * 100.
        # Probably due to different precisions the numbers between the
        # original Fortran code and the Python code differ slightly
        self.assertTrue(np.all(diffvol < 0.0318))
        self.assertTrue(np.all(diffa < 0.000722))

    def test_without_q_gradient(self):
        """
        Test forward model assuming no gradient
        of the heat input rate between time steps.
        """
        # First test with Euler
        T0 = 30.
        dt = 1.
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
        M0 = 8875.*density
        X0 = 2.
        dtime = datetime(2019, 1, 1, 0, 0, 0)
        npdp = np.dtype({'names': ['T', 'M', 'X', 'Qi', 'Mi', 'Mo', 'H', 'ws'],
                         'formats': [float] * 10})
        y0 = [T0, M0, X0, Qi0, Mi, Mo, H, ws]
        dp = [dQi, dMi, dMo, dH, dws]
        fw = Forwardmodel()
        y_new = fw.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.61, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.93, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw.get_evap()[0], 14.63, places=2)
        self.assertAlmostEqual(fw.get_evap()[1], 3.18, places=2)

        # Then test with second-order Runge-Kutta
        fw1 = Forwardmodel(method='rk2')
        y_new = fw1.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.6, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.85, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 14.9, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.25, places=2)

        # Finally test with fourth-order Runge-Kutta
        fw2 = Forwardmodel(method='rk4')
        y_new = fw2.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.56, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.86, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw2.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw2.get_evap()[0], 15.11, places=2)
        self.assertAlmostEqual(fw2.get_evap()[1], 3.31, places=2)

    def test_with_q_gradient(self):
        """
        Test forward model assuming no gradient
        of the heat input rate between time steps.
        """
        # First test with Euler
        T0 = 30.
        dt = 1.
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
        M0 = 8875.*density
        X0 = 2.
        dtime = datetime(2019, 1, 1, 0, 0, 0)
        fw = Forwardmodel()
        npdp = np.dtype({'names': ['T', 'M', 'X', 'Qi', 'Mi', 'Mo', 'H', 'ws'],
                         'formats': [float] * 10})
        y0 = [T0, M0, X0, Qi0, Mi, Mo, H, ws]
        dp = [dQi, dMi, dMo, dH, dws]
        fw = Forwardmodel()
        y_new = fw.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 30.61, places=2)
        self.assertAlmostEqual(y_new['M'], 8827.93, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw.get_evap()[0], 14.63, places=2)
        self.assertAlmostEqual(fw.get_evap()[1], 3.18, places=2)

        # Then test with second-order Runge-Kutta
        fw1 = Forwardmodel(method='rk2')
        y_new = fw1.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 31.3, places=2)
        self.assertAlmostEqual(y_new['M'], 8837.11, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 21.6, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 14.9, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.25, places=2)

        # Finally test with fourth-order Runge-Kutta
        fw2 = Forwardmodel(method='rk4')
        y_new = fw2.integrate(y0, dtime, dt, dp)
        y_new = np.array(tuple(y_new), dtype=npdp)
        self.assertAlmostEqual(y_new['T'], 31.25, places=2)
        self.assertAlmostEqual(y_new['M'], 8837.06, places=2)
        self.assertAlmostEqual(y_new['X'], 1.999, places=3)
        self.assertAlmostEqual(fw2.get_steam(), 30.86, places=2)
        self.assertAlmostEqual(fw2.get_evap()[0], 15.72, places=2)
        self.assertAlmostEqual(fw2.get_evap()[1], 3.48, places=2)

    def test_synthetic_model(self):
        """
        Test generating synthetic observations.
        """
        # Test with Euler
        df = SynModel().run(1000., nsteps=21, mode='test')
        np.testing.assert_array_almost_equal(df['T'].values,
                                             np.array([15.000, 14.953,
                                                       15.374, 16.717]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['M'].values,
                                             np.array([8782.84, 8783.277,
                                                       8786.092, 8789.457]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['X'].values,
                                             np.array([2., 1.998,
                                                       1.996, 1.993]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['Mo'].values,
                                             np.array([8.72, 9.225,
                                                       14.395, 14.395]),
                                             decimal=3)

        # Test with 4th order Runge-Kutta
        df = SynModel().run(1000., nsteps=21, mode='test',
                            integration_method='rk4')
        np.testing.assert_array_almost_equal(df['T'].values,
                                             np.array([15.000, 14.914,
                                                       15.294, 16.588]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['M'].values,
                                             np.array([8782.84, 8783.281,
                                                       8786.262, 8789.637]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['X'].values,
                                             np.array([2., 1.998,
                                                       1.996, 1.993]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['Mo'].values,
                                             np.array([8.72, 9.045,
                                                       14.328, 14.328]),
                                             decimal=3)

        # Test with 4th order Runge-Kutta and Qi gradient
        s = SynModel()
        df = s.run(1000., nsteps=21, mode='test',
                   integration_method='rk4', gradient=True)
        np.testing.assert_array_almost_equal(df['T'].values,
                                             np.array([15.000, 15.147,
                                                       15.984, 17.727]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['M'].values,
                                             np.array([8782.84, 8784.713,
                                                       8787.516, 8790.579]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['X'].values,
                                             np.array([2., 1.998,
                                                       1.995, 1.991]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(df['Mo'].values,
                                             np.array([8.72, 12.066,
                                                       17.432, 17.432]),
                                             decimal=3)

if __name__ == '__main__':
    unittest.main()
