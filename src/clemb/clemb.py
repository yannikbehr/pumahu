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

import progressbar

from clemb.forward_model import Forwardmodel


class LikeliHood:

    def __init__(self, data, date, dt, ws, cov, intmethod='rk4'):
        self.data = data
        self.date = date
        self.dt = dt
        self.ws = ws
        # the precision matrix is the inverse of the
        # covariance matrix
        self.prec = np.linalg.inv(cov)
        self.factor = -np.log(np.power(2.*np.pi, cov.shape[0])
                              + np.sqrt(np.linalg.det(cov)))
        self.samples = []
        self.fm = Forwardmodel(method=intmethod)

    def get_samples(self):
        return np.array(self.samples)

    def run_lh(self, vals, sid):
        """
        Callback function to compute the log-likelihood for nested sampling.
        """
        try:
            qi = vals[0]*0.0864
            Mi = vals[1]
            Mo = vals[2]
            H = vals[3]
            T = vals[4]
            M = vals[5]
            X = vals[6]
            dq = vals[7]
            dp = np.zeros(5)
            dp[0] = dq*0.0864
            y0 = np.array([T, M, X, qi, Mi, Mo, H, self.ws])
            y_new = self.fm.integrate(y0, self.date, self.dt, dp)
            lh = self.factor - 0.5*np.dot(y_new[0:3]-self.data,
                                          np.dot(self.prec,
                                                 y_new[0:3]-self.data))
            self.samples.append([y_new, self.fm.get_evap()[1], lh, sid])
        except:
            raise Exception("Oh no, a SamplingException!")
        return lh


class Clemb:
    """
    Compute crater lake energy and mass balance. The model currently accounts
    for evaporation effects and melt water flow inferred from the dilution of
    magnesium and chloride ions.
    """

    def __init__(self, lakedata, winddata, start, end, h=6., ws=4.5,
                 pre_txt=None, resultsd='./', save_results=True):
        """
        Load the lake data (temperature, lake level, concentration of Mg++,
        Cl-, O18 and deuterium) and the wind data.
        """
        self.lakedata = lakedata
        self.winddata = winddata
        self.h = h
        self.ws = ws
        self.pre_txt = pre_txt
        if lakedata is not None:
            self._df = lakedata.get_data(start, end).copy()
            self._dates = self._df.index
        if winddata is not None:
            self._df.loc[:, 'W'] = winddata.get_data(self._dates[0],
                                                     self._dates[-1]).copy()
            self._df.loc[:, 'H'] = np.ones(self._dates.size) * self.h
        self.use_drmg = False
        # Specific heat for water
        self.cw = 0.0042
        self.results_dir = resultsd
        self.save_results = save_results
        self.fm = Forwardmodel()
        self.fullness = self.fm.fullness

    def get_variable(self, key):
        if key in self._df:
            return self._df[key]
        elif key.lower() == 'wind':
            return self._df['W']
        elif key.lower() == 'enthalpy':
            return self._df['H']
        else:
            raise AttributeError('Unknown variable name.')

    def update_data(self, start, end):
        """
        Update the timeframe to analyse.
        """
        # allow for tinkering with the dilution factor
        old_dv = self._df['dv'].copy()

        self._df = self.lakedata.get_data(start, end).copy()
        self._dates = self._df.index
        self._df.loc[:, 'W'] = self.winddata.get_data(self._dates[0],
                                                      self._dates[-1]).copy()
        self._df.loc[:, 'H'] = np.ones(self._dates.size) * self.h

        # See if old dilution values overlap with new; if not discard
        old_start = old_dv.index.min()
        old_end = old_dv.index.max()
        new_start = self._dates.min()
        new_end = self._dates.max()
        if not old_start > new_end or not old_end < new_start:
            # Find overlapping region
            start = max(old_start, new_start)
            end = min(old_end, new_end)
            new_dv = self._df['dv'].copy()
            new_dv[start:end] = old_dv[start:end]
            self._df.loc[:, 'dv'] = new_dv

    @property
    def drmg(self):
        return self.use_drmg

    @drmg.setter
    def drmg(self, val):
        self.use_drmg = val

    def run_backward(self, new=False):
        """
        Compute the amount of steam and energy that has to be put into a crater
        lake to cause an observed temperature change. This computation runs
        the model backward using a finite difference approach.
        """
        tstart = self._dates[0]
        tend = self._dates[-1]
        res_fn = 'backward_{:s}_{:s}.nc'
        res_fn = res_fn.format(tstart.strftime('%Y-%m-%d'),
                               tend.strftime('%Y-%m-%d'))
        if self.pre_txt is not None:
            res_fn = self.pre_txt + '_' + res_fn
        res_fn = os.path.join(self.results_dir, res_fn)
        if not new and os.path.isfile(res_fn):
            res = xr.open_dataset(res_fn)
            res.close()
            return res

        ndata = self._dates.size - 1
        results = {}
        keys = ['steam', 'pwr', 'evfl', 'fmelt', 'inf', 'fmg', 'mgt', 'mg',
                'mass', 't', 'wind', 'llvl']
        for k in keys:
            results[k] = np.zeros(ndata)

        # es returns nan entries if wind is less than 0.0
        self._df.loc[self._df['W'] < 0, 'W'] = 0.0
        nd = (self._df.index[1:] - self._df.index[:-1])/pd.Timedelta('1D')
        # time interval in Megaseconds
        timem = 0.0864 * nd
        density = 1.003 - 0.00033 * self._df['T']
        a, vol = self.fullness(self._df['z'].values)
        # Dilution due to loss of water
        dr = self._df['dv'][1:].values

        mass = vol[1:] * density[1:].values
        massp = vol[:-1] * density[:-1].values

        # Dilution inferred from Mg++
        mgt = np.zeros(self._df['Mg'].size)
        drmg = np.ones(self._df['Mg'].size - 1)
        mgt[0] = massp[0] * self._df['Mg'][0]
        mgt[1:] = mass * self._df['Mg'][1:].values - \
            massp * self._df['Mg'][:-1].values / dr
        mgt = mgt.cumsum()
        fmg = np.diff(mgt) / nd
        drmg = (massp * self._df['Mg'][:-1].values /
                (mass * self._df['Mg'][1:].values))
        if self.use_drmg:
            dr = drmg

        # Net mass input to lake [kT]
        inf = massp * (dr - 1.0)  # input to replace outflow
        inf = inf + mass - massp  # input to change total mass
        loss, ev = self.fm.surface_loss(self._df['T'][:-1].values,
                                        self._df['W'][:-1].values, a[:-1])
        loss *= nd
        # Energy balances [TJ];
        # es = Surface heat loss, el = change in stored energy
        e = loss + self.el(self._df['T'][:-1].values,
                           self._df['T'][1:].values, vol[:-1])

        # e is energy required from steam, so is reduced by sun energy
        e -= self.fm.esol(nd, a[:-1], self._dates[:-1])
        # Energy = Mass * Enthalpy
        steam = e / (self._df['H'][:-1].values -
                     0.0042 * self._df['T'][:-1].values)
        evap = ev  # Evaporation loss
        meltf = inf + evap - steam  # Conservation of mass

        # Correction for energy to heat incoming meltwater
        # FACTOR is ratio: Mass of steam/Mass of meltwater (0 degrees
        # C)
        factor = self._df['T'][:-1].values * self.cw / \
            (self._df['H'][:-1].values - self._df['T'][:-1].values * self.cw)
        meltf = meltf / (1.0 + factor)  # Therefore less meltwater
        steam = steam + meltf * factor  # ...and more steam
        # Correct energy input also
        e += meltf * self._df['T'][:-1].values * self.cw

        # Flows are total amounts/day
        res = xr.Dataset({'steam': (('dates'), steam / nd),  # kT/day
                          'pwr': (('dates'), e / timem),  # MW
                          'evfl': (('dates'), evap / nd),  # kT/day
                          'fmelt': (('dates'), meltf / nd),  # kT/day
                          'mass': (('dates'), mass),
                          'inf': (('dates'), inf),
                          't': (('dates'), self._df['T'][:-1].values),
                          'fmg': (('dates'), fmg),
                          'mgt': (('dates'), mgt[:-1]),
                          'mg': (('dates'), self._df['Mg'][:-1].values),
                          'wind': (('dates'), self._df['W'][:-1].values),
                          'llvl': (('dates'), self._df['z'][:-1].values)},
                         {'dates': self._dates[:-1]})
        if self.save_results:
            res.to_netcdf(res_fn)
        return res

    def run_forward(self, nsamples=10000, nresample=500, q_in_min=0.,
                    q_in_max=1000., m_in_min=0., m_in_max=20.,
                    m_out_min=0., m_out_max=20., new=False,
                    m_out_prior=None, tolZ=1e-3, lh_fun=None,
                    tolH=3., ws=4.5, seed=-1, intmethod='rk4',
                    gradient=False):
        """
        Compute the amount of steam and energy that has to be put into a crater
        lake to cause an observed temperature change. This computation runs
        the model forward, optimising the input parameters using MCMC sampling.
        The scheme tries to optimise the misfit between the observed and
        predicted temperature, water level, and chemical concentration
        assuming normally distributed observation errors.
        """
        # Setup path for results file
        tstart = self._dates[0]
        tend = self._dates[-1]
        res_fn = 'forward_{:s}_{:s}.nc'
        res_fn = res_fn.format(tstart.strftime('%Y-%m-%d'),
                               tend.strftime('%Y-%m-%d'))
        if self.pre_txt is not None:
            res_fn = self.pre_txt + '_' + res_fn
        res_fn = os.path.join(self.results_dir, res_fn)
        if not new and os.path.isfile(res_fn):
            res = xr.open_dataset(res_fn)
            res.close()
            return res

        # setup model parameters
        nsteps = self._df.shape[0] - 1
        nparams = 8
        if gradient:
            dq = Uniform('dq', -2e3, 2e3)
        else:
            dq = Constant('dq', 0.)
        qin = Uniform('qin', q_in_min, q_in_max)
        m_in = Uniform('m_in', m_in_min, m_in_max)
        h = Constant('h', self.h)
        f_m_out_min = None
        f_m_out_max = None
        if m_out_prior is not None:
            a = np.load(m_out_prior)
            z, m_out_min, m_out_max = a['z'], a['o_min'], a['o_max']
            f_m_out_min = interp1d(z, m_out_min, fill_value='extrapolate')
            f_m_out_max = interp1d(z, m_out_max, fill_value='extrapolate')
        else:
            m_out = Uniform('m_out', m_out_min, m_out_max)

        if nresample < 0:
            nrsp = nsamples
        else:
            nrsp = nresample

        # return values
        qin_samples = np.zeros((nsteps, nrsp))*np.nan
        dqin_samples = np.zeros((nsteps, nrsp))*np.nan
        m_in_samples = np.zeros((nsteps, nrsp))*np.nan
        m_out_samples = np.zeros((nsteps, nrsp))*np.nan
        h_samples = np.zeros((nsteps, nrsp))*np.nan
        lh = np.zeros((nsteps, nrsp))*np.nan
        wt = np.zeros((nsteps, nrsp))*np.nan
        exp = np.zeros((nsteps, nparams))*np.nan
        var = np.zeros((nsteps, nparams))*np.nan
        mx = np.zeros((nsteps, nparams+1))*np.nan
        ig = np.zeros((nsteps))*np.nan
        z = np.zeros((nsteps, 2))*np.nan
        zs = np.zeros((nsteps, nrsp))*np.nan
        hs = np.zeros((nsteps, nrsp))*np.nan
        model_data = np.zeros((nsteps, nrsp, nparams))*np.nan
        mevap = np.zeros((nsteps, nrsp))*np.nan

        with progressbar.ProgressBar(max_value=nsteps-1) as bar:
            for i in range(nsteps):
                # Take samples from the input
                T = Normal('T', self._df['T'][i], self._df['T_err'][i])
                M = Normal('M', self._df['M'][i], self._df['M_err'][i])
                X = Normal('X', self._df['X'][i], self._df['X_err'][i])
                if m_out_prior is not None:
                    m_out_min = f_m_out_min(self._df['z'][i])
                    m_out_max = f_m_out_max(self._df['z'][i])
                    m_out = Uniform('m_out', float(m_out_min),
                                    float(m_out_max))
                T_sigma = self._df['T_err'][i+1]
                M_sigma = self._df['M_err'][i+1]
                X_sigma = self._df['X_err'][i+1]
                cov = np.array([[T_sigma*T_sigma, 0., 0.],
                                [0., M_sigma*M_sigma, 0.],
                                [0., 0., X_sigma*X_sigma]])
                T_next = self._df['T'][i+1]
                M_next = self._df['M'][i+1]
                X_next = self._df['X'][i+1]

                y_next = np.array([T_next, M_next, X_next])
                dt = (self._dates[i+1] - self._dates[i])/pd.Timedelta('1D')
                ns = NestedSampling(seed=seed)
                _lh = LikeliHood(data=y_next, dt=dt, ws=ws, cov=cov,
                                 date=self._dates[i], intmethod=intmethod)
                if lh_fun is None:
                    _lh_fun = _lh.run_lh
                else:
                    _lh_fun = lh_fun
                rs = ns.explore([qin, m_in, m_out, h, T, M, X, dq], 100,
                                nsamples, _lh_fun, 40, 0.1, tolZ, tolH)
                del T, M, X

                if nresample > 0:
                    smp = rs.resample_posterior(nresample)
                else:
                    smp = rs.get_samples()

                exp[i, :] = rs.getexpt()
                var[i, :] = rs.getvar()
                mx[i, :] = rs.getmax()
                ig[i] = rs.getH()
                z[i, :] = rs.getZ()
                lh_samples = _lh.get_samples()
                for j, _s in enumerate(smp):
                    Q_in, M_in, M_out, H, T, M, X, dQ_in = _s.get_value()
                    sid = np.where(lh_samples[:, -1] == _s.get_id())
                    y_mod, me, _, _ = lh_samples[sid][0]
                    mevap[i, j] = me
                    model_data[i, j, :] = y_mod[:]
                    qin_samples[i, j] = Q_in
                    dqin_samples[i, j] = dQ_in
                    m_in_samples[i, j] = M_in
                    m_out_samples[i, j] = M_out
                    h_samples[i, j] = H
                    lh[i, j] = np.exp(_s.get_logL())
                    wt[i, j] = np.exp(_s.get_logWt())
                    zs[i, j] = _s.get_logZ()
                    hs[i, j] = _s.get_H()

                del smp, ns, rs
                bar.update(i)

        res = xr.Dataset({'exp': (('dates', 'parameters'), exp),
                          'var': (('dates', 'parameters'), var),
                          'max': (('dates', 'parameters'), mx[:, :-1]),
                          'z': (('dates', 'val_std'), z),
                          'ig': (('dates'), ig),
                          'q_in': (('dates', 'sampleidx'), qin_samples),
                          'dq_in': (('dates', 'sampleidx'), dqin_samples),
                          'h': (('dates', 'sampleidx'), h_samples),
                          'lh': (('dates', 'sampleidx'), lh),
                          'wt': (('dates', 'sampleidx'), wt),
                          'zs': (('dates', 'sampleidx'), zs),
                          'hs': (('dates', 'sampleidx'), hs),
                          'm_in': (('dates', 'sampleidx'), m_in_samples),
                          'm_out': (('dates', 'sampleidx'), m_out_samples),
                          'mevap': (('dates', 'sampleidx'), mevap),
                          'model': (('dates_p', 'sampleidx', 'obs'),
                                    model_data)},
                         {'dates': self._dates[:-1].values,
                          'dates_p': self._dates[1:].values,
                          'parameters': ['q_in', 'm_in', 'm_out',
                                         'h', 'T', 'M', 'X', 'dq_in'],
                          'obs': ['T', 'M', 'X', 'q_in', 'm_in',
                                  'm_out', 'h', 'ws'],
                          'val_std': ['val', 'std']})
        if self.save_results:
            res.to_netcdf(res_fn)
        return res

    def el(self, t1, t2, vol):
        """
        Change in Energy stored in the lake [TJ]
        """
        return (t2 - t1) * vol * self.cw
