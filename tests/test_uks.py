from collections import OrderedDict
import numpy as np
import unittest
import pandas as pd
import xarray as xr

from filterpy.kalman import unscented_transform

from pumahu import get_data
from pumahu.uks import UnscentedKalmanSmoother
from pumahu.syn_model import (SynModel,
                              setup_test,
                              setup_realistic)
from pumahu.data import LakeData
from pumahu.sigma_points import MerweScaledSigmaPoints


class UKSTestCase(unittest.TestCase):

    def setUp(self):
        self.xds = SynModel(integration_method='rk4').run(setup_test(),
                                                          gradient=True)
        self.Q = OrderedDict(T=1e-3, M=1e-3, X=1e-3, q_in=1e-10,
                             m_in=1e1, m_out=1e1, h=1e-3, W=1e1,
                             dqi=1e3, dMi=1e4, dMo=1e4, dH=1e-3, dW=1e1)
        self.Q = np.eye(len(self.Q))*list(self.Q.values())

        self.P0 = OrderedDict(T=1e3, M=1e3, X=1e3, q_in=1e3,
                              m_in=1e3, m_out=1e3, h=1e-1, W=1e-1,
                              dqi=1e3, dMi=1e3, dMo=1e3, dH=1e-1, dW=1e-1)
        self.P0 = np.eye(len(self.P0))*list(self.P0.values())

        self.xds_nan = self.xds.copy(deep=True)
        self.xds_nan.exp[2, 0, 0] = np.nan
        self.xds_nan.exp[2, 0, 1] = np.nan

    def test_init(self):
        """
        Test that the init function works correctly
        """
        uks = UnscentedKalmanSmoother(data=self.xds.exp)
        self.assertEqual(len(uks.X0), uks.nvar)
        self.assertTrue(isinstance(uks.X0, list))

    def test_synthetic(self):
        uks = UnscentedKalmanSmoother(data=self.xds.exp)
        res = uks(test=True)
        log_lh = np.nansum(res['p_samples'].loc[dict(p_parameters='lh')].values)
        self.assertAlmostEqual(log_lh, -29.2659392, 4) 
        
    def test_synthetic_extensive(self):
        """
        Test with more extensive synthetic model.
        """
        xds3 = SynModel().run(setup_realistic(sinterval=24*60))
        Q = OrderedDict(T=1e-3, M=1e-3, X=1e-3, q_in=1e0,
                        m_in=1e1, m_out=1e1, h=1e-10, W=1e1,
                        dqi=5e3, dMi=1e4, dMo=1e4, dH=1e-3, 
                        dW=1e1)
        Q = np.eye(len(Q))*list(Q.values())
        P0 = OrderedDict(T=1e0, M=1e0, X=1e0, q_in=1e3,
                         m_in=1e3, m_out=1e3, h=1e-1, W=1e-1,
                         dqi=1e3, dMi=1e3, dMo=1e3, dH=1e-1, 
                         dW=1e-1)
        P0 = np.eye(len(P0))*list(P0.values())
        uks = UnscentedKalmanSmoother(data=xds3.exp, P0=P0, Q=Q)
        rs = uks(test=True, smooth=True)
        logl_mean = np.nansum(rs.p_samples.values)/rs.dates.shape[0]
        self.assertAlmostEqual(logl_mean, -12.56475, 4)

    def test_nans(self):
        """
        Test NaN's in input.
        """
        uks = UnscentedKalmanSmoother(data=self.xds_nan.exp)
        rs = uks(test=True)
        logl = rs.p_samples[:, 0]
        np.testing.assert_array_almost_equal(logl,
                                             np.array([np.nan,
                                                       -12.0177,
                                                       -8.8329,
                                                       -7.8945]), 4)

    def test_real_data(self):
        ld = LakeData()
        data = ld.get_data('2019-01-01', '2019-02-01', smoothing='dv')
        ld.get_outflow()
        uks = UnscentedKalmanSmoother(data=data)
        rs = uks(test=True)
        logl_mean = np.nansum(rs.p_samples.values)/rs.dates.shape[0]
        self.assertAlmostEqual(logl_mean, -6.40177, 4)
        
    def test_historic_data(self):
        """
        Test with dataset used by Hurst et al. [2015]. 
        """
        ld = LakeData(csvfile=get_data('data/data.csv'))
        xdf = ld.get_data('2000-1-1', '2021-1-1',
                          smoothing={'Mg': 2.6, 'T': 0.4, 'z': 0.01})
        xdf = xdf.dropna('dates', how='all')
        P0 = OrderedDict(T=1e0, M=1e0, X=1e0, q_in=1e1,
                         m_in=1e3, m_out=1e3, h=1e-1, W=1e-1,
                         dqi=1e-1, dMi=1e-1, dMo=1e-1, dH=1e-1, 
                         dW=1e-1)
        P0 = np.eye(len(P0))*list(P0.values())

        Q = OrderedDict(T=1e-1, M=1e1, X=1e-3, q_in=1e1,
                        m_in=1e1, m_out=1e1, h=1e-3, W=1e-3,
                        dqi=0, dMi=0, dMo=0, dH=0, dW=0)
        Q = np.eye(len(Q))*list(Q.values())
        uks = UnscentedKalmanSmoother(data=xdf, Q=Q, P0=P0)
        xds_uks = uks() 

    def test_truncated_sigma_pts(self):
        """
        Test the interval-constrained UKF based on Teixeira et al.,
        Journal of Process Control, 2010
        """
        x = np.array([1., 1.])
        Px = np.eye(2)
        points = MerweScaledSigmaPoints(2, 1., 1., 0.)
        test_points = np.array([[1., 1.],
                                [2.41421356, 1.],
                                [1., 1.75],
                                [0., 1.],
                                [1., -0.41421356]])
        test_weights = np.array([[0.10816111, 0.25, 0.18338254,
                                  0.20845635, 0.25],
                                 [1.10816111, 0.25, 0.18338254,
                                  0.20845635, 0.25]])
        w1, s1 = points.constrained_sigma_points(x, Px,
                                                 np.array([3, 3]),
                                                 np.array([-1, -1.]))
        w2, s2 = points.sigma_points(x, Px)
        x2, P2 = unscented_transform(s2, w2[0], w2[1])
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_almost_equal(w1[0], points.Wm, 6)
        np.testing.assert_array_almost_equal(w1[1], points.Wc, 6)
        e = np.array([3, 1.75])
        d = np.array([0, -1.])
        w3, s3 = points.constrained_sigma_points(x, Px, e, d)
        x3, P3 = unscented_transform(s3, w3[0], w3[1])
        np.testing.assert_array_almost_equal(test_points, s3, 6)
        np.testing.assert_array_almost_equal(test_weights, w3, 6)
        w4, s4 = points.constrained_sigma_points(np.array([-1, -1]), Px,
                                                 e, d)
        x4, P4 = unscented_transform(s4, w4[0], w4[1])
        np.testing.assert_array_less(np.full(s4.shape, -1.1), s4)

        s5 = np.where(s2 < d, d, s2)
        s5 = np.where(s5 > e, e, s5)
        x5, P5 = unscented_transform(s5, w2[0], w2[1])

        if False:
            import matplotlib.pyplot as plt
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            plt.figure(figsize=(6,6))
            theta = np.linspace(0, 2*np.pi, 100)
            plt.scatter(s2[:,0], s2[:,1], marker='^', color=cycle[0])
            plt.scatter(x2[0], x2[1], marker='o', color=cycle[0])
            plt.plot(x2[0] + np.sqrt(P2[0,0])*np.cos(theta),
                     x2[1] + np.sqrt(P2[1,1])*np.sin(theta),
                     color=cycle[0])
            plt.scatter(s3[:,0], s3[:,1], marker='^', color=cycle[1])
            plt.scatter(x3[0], x3[1], marker='o', color=cycle[1])
            plt.plot(x3[0] + np.sqrt(P3[0,0])*np.cos(theta),
                     x3[1] + np.sqrt(P3[1,1])*np.sin(theta),
                     color=cycle[1])
            plt.scatter(s4[:,0], s4[:,1], marker='^', color=cycle[2])
            plt.scatter(x4[0], x4[1], marker='o', color=cycle[2])
            plt.plot(x4[0] + np.sqrt(P4[0,0])*np.cos(theta),
                     x4[1] + np.sqrt(P4[1,1])*np.sin(theta),
                     color=cycle[2])
            plt.scatter(s5[:,0], s5[:,1], marker='^', color=cycle[3])
            plt.scatter(x5[0], x5[1], marker='o', color=cycle[3])
            plt.plot(x5[0] + np.sqrt(P5[0,0])*np.cos(theta),
                     x5[1] + np.sqrt(P5[1,1])*np.sin(theta),
                     color=cycle[3])
            plt.hlines(-1, -1, 3.5, linestyle='--', color='k')
            plt.hlines(1.75, -1, 3.5, linestyle='--', color='k')
            plt.vlines(0, -1.5, 3, linestyle='--', color='k')
            plt.vlines(3, -1.5, 3, linestyle='--', color='k')
            plt.xlim(-1,3.5)
            plt.ylim(-1.5,3)
            plt.show()

 
if __name__ == '__main__':
    unittest.main()
