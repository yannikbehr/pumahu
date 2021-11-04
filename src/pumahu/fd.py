"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""

import os

import numpy as np
from numpy.ma import masked_invalid, masked_less
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.integrate import trapz, cumtrapz
import xarray as xr

from nsampling import (NestedSampling, Uniform,
                       Normal, InvCDF, Constant)

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_continuous_white_noise

from pumahu.forward_model import Forwardmodel


def el(t1, t2, vol, cw):
    """
    Change in Energy stored in the lake [TJ]
    """
    return (t2 - t1) * vol * cw


def fd(data, results_file=None, new=False, use_drmg=False,
       level2volume=None):
    """
    Compute the amount of steam and energy that has to be put into a crater
    lake to cause an observed temperature change. This computation runs
    the model backward using a finite difference approach.
    """
    if results_file is not None:
        if not new and os.path.isfile(results_file):
            res = xr.open_dataset(results_file)
            res.close()
            return res
    parameters = ['q_in', 'm_in', 'm_out',
                  'h', 'T', 'M', 'X', 'dq_in']
    nparams = len(parameters)
    dates = pd.to_datetime(data['dates'].values)
    ndates = dates.size
    exp = np.zeros((ndates, nparams, 2))*np.nan
    
    data_z = data.loc[:, 'z', 'val'].values
    data_T = data.loc[:, 'T', 'val'].values
    data_mg = data.loc[:, 'Mg', 'val'].values
    data_W = data.loc[:, 'W', 'val'].values
    data_h = data.loc[:, 'h', 'val'].values
    fm = Forwardmodel(level2volume=level2volume)
    cw=0.0042

    # es returns nan entries if wind is less than 0.0
    nd = (dates[1:] - dates[:-1])/pd.Timedelta('1D')
    # time interval in Megaseconds
    timem = 0.0864 * nd
    density = 1.003 - 0.00033 * data_T
    a, vol = fm.level2volume(data_z)
    # Dilution due to loss of water
    dr = np.ones(ndates-1)

    mass = vol[1:] * density[1:]
    massp = vol[:-1] * density[:-1]

    # Dilution inferred from Mg++
    drmg = (massp * data_mg[:-1] /
            (mass * data_mg[1:]))
    if use_drmg:
        dr = drmg

    m_out_idx = parameters.index('m_out')
    exp[:-1, m_out_idx, 0] = (dr - 1.0) * massp / nd 
    exp[:-1, m_out_idx, 1] = np.zeros(ndates-1) 

    # Net mass input to lake [kT]
    inf = massp * (dr - 1.0)  # input to replace outflow
    inf = inf + mass - massp  # input to change total mass

    loss, ev = fm.surface_loss(data_T[:-1], data_W[:-1], a[:-1])
    loss *= nd
    # Energy balances [TJ];
    # es = Surface heat loss, el = change in stored energy
    e = loss + el(data_T[:-1], data_T[1:], vol[:-1], cw)

    # e is energy required from steam, so is reduced by sun energy
    e -= nd * fm.esol(a[:-1], dates[:-1])
    # Energy = Mass * Enthalpy
    steam = e / (data_h[:-1] - cw * data_T[:-1])
    meltf = inf + ev - steam  # Conservation of mass

    # Correction for energy to heat incoming meltwater
    # FACTOR is ratio: Mass of steam/Mass of meltwater (0 degrees
    # C)
    factor = data_T[:-1] * cw / (data_h[:-1] - data_T[:-1] * cw)
    meltf = meltf / (1.0 + factor)  # Therefore less meltwater
    steam = steam + meltf * factor  # ...and more steam
    # Correct energy input also
    e += meltf * data_T[:-1] * cw

    m_in_idx = parameters.index('m_in')
    exp[:-1, m_in_idx, 0] = meltf / nd 
    exp[:-1, m_in_idx, 1] = np.zeros(ndates-1) 


    q_in_idx = parameters.index('q_in')
    exp[:-1, q_in_idx, 0] = e / timem 
    exp[:-1, q_in_idx, 1] = np.zeros(ndates-1) 

    M_idx = parameters.index('M')
    exp[:-1, M_idx, 0] = mass 
    exp[:-1, M_idx, 1] = np.zeros(ndates-1) 

      # Save results to disk
    res = xr.Dataset({'exp': (('dates', 'parameters', 'val_std'), exp),
                      'input': (('dates', 'i_parameters', 'val_std'),
                                data.values)},
                     {'dates': dates.values,
                      'parameters': parameters,
                      'i_parameters': data['parameters'].values,
                      'val_std': ['val', 'std']})

    if results_file is not None:
        res.to_netcdf(results_file)
    return res