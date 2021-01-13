import numpy as np
import unittest
import pandas as pd
import xarray as xr

from filterpy.kalman import unscented_transform

from .uks import UnscentedKalmanSmoother
from .syn_model import SynModel
from .data import LakeData
from .sigma_points import MerweScaledSigmaPoints


class UKSTestCase(unittest.TestCase):

    def setUp(self):
        self.xds = SynModel(seed=42).run(1000., nsteps=100,
                                         integration_method='rk4',
                                         gradient=True, mode='gamma',
                                         addnoise=True,
                                         estimatenoise=True)
        self.df = self.xds.to_dataframe()
        self.dt = (self.df.index[1] - self.df.index[0])/pd.Timedelta('1D')
        self.Q = np.eye(9)*[1e-3, 1e-3, 1e-3, 1e-10,
                            1e1, 1e1, 1e3, 1e4, 1e4]*self.dt**2
        (T, M, X, self.Mi,
         self.Mo, self.qi) = self.df.iloc[0][['T', 'M', 'X',
                                              'Mi', 'Mo', 'qi']]
        self.dqi = 1e-1
        self.dMi = 1e-1
        self.dMo = 1e-1
        self.X0 = [T, M, X, self.qi*0.0864, self.Mi, self.Mo,
                   self.dqi, self.dMi, self.dMo]
        self.P0 = np.eye(9)*1e3

        npts = self.df.shape[0]
        self.df_nan = self.df.copy()
        self.df_nan['X'][1:npts] = np.nan
        self.df_nan['X_err'][1:npts] = np.nan

        self.df_nan2 = self.df.copy()
        self.df_nan2['X'][:] = np.nan
        self.df_nan2['X_err'][:] = np.nan

    def test_synthetic(self):
        uks = UnscentedKalmanSmoother(data=self.df)
        log_lh = uks(self.Q, self.X0, self.P0, test=True)
        self.assertAlmostEqual(log_lh, 12.8047, 4)

    def test_nans(self):
        """
        Test NaN's in input.
        """
        uks = UnscentedKalmanSmoother(data=self.df_nan)
        log_lh = uks(self.Q, self.X0, self.P0, test=True)

        uks1 = UnscentedKalmanSmoother(data=self.df_nan2)
        log_lh1 = uks1(self.Q, self.X0, self.P0, test=True)
        self.assertAlmostEqual(log_lh, -13.5423, 4)
        self.assertAlmostEqual(log_lh1, -13.5423, 4)

    def test_likelihood(self):
        uks = UnscentedKalmanSmoother(data=self.df)
        q = np.diag(self.Q)/self.dt**2
        x0 = np.array([self.qi, self.Mi, self.Mo,
                       self.dqi, self.dMi, self.dMo])
        log_lh = uks.likelihood(np.r_[q, x0, 1e3], 1)
        self.assertAlmostEqual(log_lh, 12.8047, 4)

    def test_real_data(self):
        ld = LakeData()
        df_rd = ld.get_data_fits('2019-01-01', '2020-02-01', smoothing='dv')
        ld.get_outflow()
        ld.get_MetService_wind('/home/behrya/GeoNet/wind', default=None)
        var_names = ['T', 'M', 'X', 'qi', 'Mi',
                     'Mo', 'H', 'W', 'dqi',
                     'dMi', 'dMo', 'dH', 'dW']
        nvars = len(var_names)

        X0 = xr.DataArray(np.ones(nvars)*np.nan, dims=('x'),
                          coords={'x': var_names})
        for k in var_names[:3]:
            X0.loc[k] = df_rd.iloc[0][k]
        X0.loc['qi'] = 100
        X0.loc['Mi'] = 10.
        X0.loc['Mo'] = 10.
        X0.loc['H'] = 3.
        X0.loc['W'] = 4.5
        X0.loc['dqi'] = 1e-1
        X0.loc['dMi'] = 1e-1
        X0.loc['dMo'] = 1e-1
        X0.loc['dW'] = 0
        X0.loc['dH'] = 0
        X0.loc['qi'] *= 0.0864

        Q = xr.DataArray(np.eye(nvars, nvars), dims=('x', 'y'),
                         coords={'x': var_names, 'y': var_names})
        Q.loc['T', 'T'] = 1e-2
        Q.loc['M', 'M'] = 1e0
        Q.loc['X', 'X'] = 1e-3
        Q.loc['qi', 'qi'] = 1e-10
        Q.loc['Mi', 'Mi'] = 1
        Q.loc['Mo', 'Mo'] = 1
        Q.loc['W', 'W'] = 1e-10
        Q.loc['H', 'H'] = 1e0
        Q.loc['dqi', 'dqi'] = 1
        Q.loc['dMi', 'dMi'] = 1e2
        Q.loc['dMo', 'dMo'] = 1
        Q.loc['dW', 'dW'] = 1e-3
        Q.loc['dH', 'dH'] = 1e-3

        P0 = xr.DataArray(np.eye(nvars, nvars), dims=('x', 'y'),
                          coords={'x': var_names, 'y': var_names})
        P0.loc['T', 'T'] = .1*X0.loc['T']
        P0.loc['M', 'M'] = .1*X0.loc['M']
        P0.loc['X', 'X'] = .1*X0.loc['X']
        P0.loc['qi', 'qi'] = 100*0.0864
        P0.loc['Mi', 'Mi'] = 1e2
        P0.loc['Mo', 'Mo'] = 1e2
        P0.loc['W', 'W'] = 1e-10
        P0.loc['H', 'H'] = 1e1
        P0.loc['dqi', 'dqi'] = 1
        P0.loc['dMi', 'dMi'] = 1
        P0.loc['dMo', 'dMo'] = 1
        P0.loc['dW', 'dW'] = 1e-10
        P0.loc['dH', 'dH'] = 1
        uks = UnscentedKalmanSmoother(data=df_rd)
        log_lh = uks(Q, X0, P0, test=True)
        print(log_lh)

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

    def test_constrained_uks_synthetic(self):
        xds = SynModel(seed=42).run(1000., nsteps=100,
                                    integration_method='rk4',
                                    gradient=True, mode='gamma+sin',
                                    addnoise=True,
                                    estimatenoise=False)

        np.random.seed(55)
        gap_idx = np.random.randint(0, xds['Mo'].size, int(0.2*xds['Mo'].size))
        a = np.arange(0, 100, 5)
        b = np.arange(100)
        gap_idx = np.setdiff1d(b, a)
        xds['Mo'][gap_idx] = np.nan
        xds['Mo_err'][gap_idx] = np.nan
        xds['X'][gap_idx] = np.nan
        xds['X_err'][gap_idx] = np.nan
        xds['W_err'] *= 1e0

        data = xds.to_dataframe()
        var_names = ['T', 'M', 'X', 'qi', 'Mi',
                     'Mo', 'H', 'W', 'dqi',
                     'dMi', 'dMo', 'dH', 'dW']
        nvars = len(var_names)

        X0 = xr.DataArray(np.ones(nvars)*np.nan, dims=('x'),
                          coords={'x': var_names})
        for k in var_names[:8]:
            X0.loc[k] = data.iloc[0][k]
        X0.loc['dqi'] = 1e-1
        X0.loc['dMi'] = 1e-1
        X0.loc['dMo'] = 1e-1
        X0.loc['dW'] = 0
        X0.loc['dH'] = 0
        X0.loc['qi'] *= 0.0864

        Q = xr.DataArray(np.eye(nvars, nvars), dims=('x', 'y'),
                         coords={'x': var_names, 'y': var_names})
        Q.loc['T', 'T'] = 1e-3
        Q.loc['M', 'M'] = 1e-3
        Q.loc['X', 'X'] = 1e-3
        Q.loc['qi', 'qi'] = 1e2
        Q.loc['Mi', 'Mi'] = 1e1
        Q.loc['Mo', 'Mo'] = 1e1
        Q.loc['W', 'W'] = 1e1
        Q.loc['H', 'H'] = 1e1
        Q.loc['dqi', 'dqi'] = 1e3
        Q.loc['dMi', 'dMi'] = 1e4
        Q.loc['dMo', 'dMo'] = 1e4
        Q.loc['dW', 'dW'] = 1e1
        Q.loc['dH', 'dH'] = 1e1

        P0 = xr.DataArray(np.eye(nvars, nvars), dims=('x', 'y'),
                          coords={'x': var_names, 'y': var_names})
        P0.loc['T', 'T'] = .1*X0.loc['T']
        P0.loc['M', 'M'] = .1*X0.loc['M']
        P0.loc['X', 'X'] = .5*X0.loc['X']
        P0.loc['qi', 'qi'] = 100*0.0864
        P0.loc['Mi', 'Mi'] = 1e2
        P0.loc['Mo', 'Mo'] = 1e2
        P0.loc['W', 'W'] = 1e1
        P0.loc['H', 'H'] = 1e1
        P0.loc['dqi', 'dqi'] = 1
        P0.loc['dMi', 'dMi'] = 1
        P0.loc['dMo', 'dMo'] = 1
        P0.loc['dW', 'dW'] = 1
        P0.loc['dH', 'dH'] = 1

        uks = UnscentedKalmanSmoother(data=xds.to_dataframe())
        log_lh = uks(Q, X0, P0, test=True)
        print(log_lh)


if __name__ == '__main__':
    unittest.main()
