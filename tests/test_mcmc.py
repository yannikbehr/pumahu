from collections import defaultdict
import inspect
import os
import unittest
import warnings

import numpy as np
import pandas as pd
import pytest

from pumahu.data import LakeData, WindData
from pumahu.mcmc import ns_sampling
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
                                             np.array([163.982627,
                                                       294.531580,
                                                       620.161183,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(q_in_var,
                                             np.array([12335.374329,
                                                       38283.162623,
                                                       68156.742499,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_val,
                                             np.array([-9.892624,
                                                       -9.829991,
                                                       -9.863981,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_var,
                                             np.array([0.180277,
                                                       0.174511,
                                                       0.171636,
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
                                             np.array([157.990046,
                                                       315.865505,
                                                       650.040960,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(q_in_var,
                                             np.array([10232.987854,
                                                       43354.119170,
                                                       68017.325559,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_val,
                                             np.array([-9.600456,
                                                       -9.604083,
                                                       -9.882427,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_var,
                                             np.array([0.176571,
                                                       0.164973,
                                                       0.1727552,
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
                                             np.array([162.698348,
                                                       339.231364,
                                                       614.788254,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(q_in_var,
                                             np.array([11269.343358,
                                                       39485.261664,
                                                       66673.434449,
                                                       np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_val,
                                             np.array([-9.645880,
                                                       -9.674900,
                                                       -9.774130,
                                                        np.nan]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(z_var,
                                             np.array([0.176319,
                                                       0.169456,
                                                       0.167194,
                                                       np.nan]),
                                             decimal=6)


if __name__ == '__main__':
    unittest.main()
