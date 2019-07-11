import unittest

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
        v = 8875
        Qi0 = 400. * 0.0864
        dQi = 0.
        M_melt = 10
        M_out = 5.
        H = 2.8
        ws = 4.5
        density = 1.003 - 0.00033 * T0
        M0 = v*density
        X0 = 2.
        y0 = [T0, M0, X0, Qi0]
        fw = Forwardmodel()
        solar = fw.esol(dt, a, 1)
        y_new = fw.integrate(y0, dt, dQi, a, v, M_melt, M_out, solar, H, ws)
        T1, M1, X1, Qi1 = y_new
        self.assertAlmostEqual(T1, 30.61, places=2)
        self.assertAlmostEqual(M1, 8827.96, places=2)
        self.assertAlmostEqual(X1, 1.999, places=3)
        self.assertAlmostEqual(fw.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw.get_evap()[0], 14.48, places=2)
        self.assertAlmostEqual(fw.get_evap()[1], 3.15, places=2)

        # Then test with second-order Runge-Kutta
        fw1 = Forwardmodel(method='rk2')
        y_new = fw1.integrate(y0, dt, dQi, a, v, M_melt, M_out, solar, H, ws)
        T1, M1, X1, Qi1 = y_new
        self.assertAlmostEqual(T1, 30.6, places=2)
        self.assertAlmostEqual(M1, 8827.89, places=2)
        self.assertAlmostEqual(X1, 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 14.73, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.22, places=2)

        # Finally test with fourth-order Runge-Kutta
        fw1 = Forwardmodel(method='rk4')
        y_new = fw1.integrate(y0, dt, dQi, a, v, M_melt, M_out, solar, H, ws)
        T1, M1, X1, Qi1 = y_new
        self.assertAlmostEqual(T1, 30.6, places=2)
        self.assertAlmostEqual(M1, 8827.89, places=2)
        self.assertAlmostEqual(X1, 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 14.98, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.28, places=2)

    def test_with_q_gradient(self):
        """
        Test forward model assuming no gradient
        of the heat input rate between time steps.
        """
        # First test with Euler
        T0 = 30.
        dt = 1.
        a = 194162
        v = 8875
        Qi0 = 400. * 0.0864
        dQi = (1000. - 400.) * 0.0864
        M_melt = 10
        M_out = 5.
        H = 2.8
        ws = 4.5
        density = 1.003 - 0.00033 * T0
        M0 = v*density
        X0 = 2.
        y0 = [T0, M0, X0, Qi0]
        fw = Forwardmodel()
        solar = fw.esol(dt, a, 1)
        y_new = fw.integrate(y0, dt, dQi, a, v, M_melt, M_out, solar, H, ws)
        T1, M1, X1, Qi1 = y_new
        self.assertAlmostEqual(T1, 30.61, places=2)
        self.assertAlmostEqual(M1, 8827.96, places=2)
        self.assertAlmostEqual(X1, 1.999, places=3)
        self.assertAlmostEqual(fw.get_steam(), 12.34, places=2)
        self.assertAlmostEqual(fw.get_evap()[0], 14.48, places=2)
        self.assertAlmostEqual(fw.get_evap()[1], 3.15, places=2)

        # Then test with second-order Runge-Kutta
        fw1 = Forwardmodel(method='rk2')
        y_new = fw1.integrate(y0, dt, dQi, a, v, M_melt, M_out, solar, H, ws)
        T1, M1, X1, Qi1 = y_new
        self.assertAlmostEqual(T1, 31.3, places=2)
        self.assertAlmostEqual(M1, 8837.15, places=2)
        self.assertAlmostEqual(X1, 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 21.6, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 14.73, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.22, places=2)

        # Finally test with fourth-order Runge-Kutta
        fw1 = Forwardmodel(method='rk4')
        y_new = fw1.integrate(y0, dt, dQi, a, v, M_melt, M_out, solar, H, ws)
        T1, M1, X1, Qi1 = y_new
        self.assertAlmostEqual(T1, 31.29, places=2)
        self.assertAlmostEqual(M1, 8837.09, places=2)
        self.assertAlmostEqual(X1, 1.999, places=3)
        self.assertAlmostEqual(fw1.get_steam(), 30.86, places=2)
        self.assertAlmostEqual(fw1.get_evap()[0], 15.58, places=2)
        self.assertAlmostEqual(fw1.get_evap()[1], 3.45, places=2)




if __name__ == '__main__':
    unittest.main()
