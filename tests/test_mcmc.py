from collections import defaultdict
import inspect
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import pytest

from pumahu.data import LakeData, WindData
from pumahu.mcmc import ns_sampling, main
from pumahu.syn_model import SynModel, setup_test


class MCMCTestCase(unittest.TestCase):

    @pytest.mark.slow
    def test_synthetic_euler(self):
        data = SynModel(integration_method='euler').run(setup_test())
        rs = ns_sampling(data.exp, nsamples=2000, nresample=-1,
                         q_in_lim=(0., 1500.), m_in_lim=(0., 40.),
                         m_out_lim=(0., 40.), new=True, tolZ=1e-3, 
                         tolH=3e30, H=6., ws=4.5, seed=42, intmethod='euler',
                         gradient=False)
        q_in_val = rs['exp'].loc[:, 'q_in', 'val'].data
        q_in_var = rs['exp'].loc[:, 'q_in', 'std'].data
        z_val = rs.p_samples.loc[:, :, 'Z'].max(axis=1).data 
        z_var = rs.p_samples.loc[:, :, 'Z_var'].max(axis=1).data 
        np.testing.assert_array_almost_equal(q_in_val,
                                             np.array([150.20788797,
                                                       308.42346672,
                                                       626.16570884,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(q_in_var,
                                             np.array([10452.13754637,
                                                       39601.25850057,
                                                       54910.75889361,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_val,
                                             np.array([-9.54064536,
                                                       -9.92588325,
                                                       -9.33493868,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_var,
                                             np.array([0.17426234,
                                                       0.17471695,
                                                       0.1573276,
                                                       np.nan]),
                                             decimal=6)

    @pytest.mark.slow
    def test_synthetic_rk2(self):
        """
        Test the inversion of a synthetic model using fourth-order
        Runge-Kutta integration.
        """
        data = SynModel(integration_method='rk2').run(setup_test())
        rs = ns_sampling(data.exp, nsamples=2000, nresample=-1,
                         q_in_lim=(0., 1500.), m_in_lim=(0., 40.),
                         m_out_lim=(0., 40.), new=True, tolZ=1e-3, 
                         tolH=3e30, H=6., ws=4.5, seed=42, intmethod='rk2',
                         gradient=False)
        q_in_val = rs['exp'].loc[:, 'q_in', 'val'].data
        q_in_var = rs['exp'].loc[:, 'q_in', 'std'].data
        z_val = rs.p_samples.loc[:, :, 'Z'].max(axis=1).data 
        z_var = rs.p_samples.loc[:, :, 'Z_var'].max(axis=1).data 
        np.testing.assert_array_almost_equal(q_in_val,
                                             np.array([146.429074,
                                                       323.870525,
                                                       602.184093,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(q_in_var,
                                             np.array([9803.695870,
                                                       32647.285447,
                                                       57228.341571,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_val,
                                             np.array([-9.580015,
                                                       -9.595466,
                                                       -9.577152,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_var,
                                             np.array([0.177103,
                                                       0.167645,
                                                       0.162358,
                                                       np.nan]),
                                             decimal=6)

    @pytest.mark.slow
    def test_synthetic_rk4(self):
        """
        Test the inversion of a synthetic model using fourth-order
        Runge-Kutta integration.
        """
        data = SynModel(integration_method='rk4').run(setup_test())
        rs = ns_sampling(data.exp, nsamples=2000, nresample=-1,
                         q_in_lim=(0., 1500.), m_in_lim=(0., 40.),
                         m_out_lim=(0., 40.), new=True, tolZ=1e-3, 
                         tolH=3e30, H=6., ws=4.5, seed=42, intmethod='rk4',
                         gradient=False)
        q_in_val = rs['exp'].loc[:, 'q_in', 'val'].data
        q_in_var = rs['exp'].loc[:, 'q_in', 'std'].data
        z_val = rs.p_samples.loc[:, :, 'Z'].max(axis=1).data 
        z_var = rs.p_samples.loc[:, :, 'Z_var'].max(axis=1).data 
        np.testing.assert_array_almost_equal(q_in_val,
                                             np.array([143.67839003,
                                                       315.51390907,
                                                       614.56922148,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(q_in_var,
                                             np.array([10255.19277908,
                                                       37474.98777957,
                                                       49952.45786385,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_val,
                                             np.array([-9.61335748,
                                                       -9.555548,
                                                       -9.38333698,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_var,
                                             np.array([0.17608661,
                                                       0.16448248,
                                                       0.16198946,
                                                       np.nan]),
                                             decimal=6)

    def test_main(self):
        rdir = tempfile.gettempdir()
        main(['-p', '-f', '--rdir', rdir,
             '-s', '20210801', '-e', '20210818'])


if __name__ == '__main__':
    unittest.main()
