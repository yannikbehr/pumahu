import numpy as np
import unittest
import pandas as pd

from clemb.uks import UnscentedKalmanSmoother
from clemb.syn_model import SynModel


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


if __name__ == '__main__':
    unittest.main()
