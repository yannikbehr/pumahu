"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""

from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xarray as xr
import progressbar

from nsampling import (NestedSampling, Uniform,
                       Normal, InvCDF, Constant)

from clemb.forward_model import Forwardmodel
from clemb.data import LakeData


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

    
def ns_sampling(data, results_file, nsamples=10000, nresample=500,
                q_in_min=0., q_in_max=1000., m_in_min=0., m_in_max=20.,
                m_out_min=0., m_out_max=20., new=False,
                m_out_prior=None, tolZ=1e-3, lh_fun=None,
                tolH=3., H=6., ws=4.5, seed=-1, intmethod='rk4',
                gradient=False):
    """
    Compute the amount of steam and energy that has to be put into a crater
    lake to cause an observed temperature change. This computation runs
    the model forward, optimising the input parameters using MCMC sampling.
    The scheme tries to optimise the misfit between the observed and
    predicted temperature, water level, and chemical concentration
    assuming normally distributed observation errors.
    """
    dates = data.index
    if not new and os.path.isfile(results_file):
        res = xr.open_dataset(results_file)
        res.close()
        return res

    # setup model parameters
    nsteps = data.shape[0] - 1
    nparams = 8
    if gradient:
        dq = Uniform('dq', -2e3, 2e3)
    else:
        dq = Constant('dq', 0.)
    qin = Uniform('qin', q_in_min, q_in_max)
    m_in = Uniform('m_in', m_in_min, m_in_max)
    h = Constant('h', H)
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
            T = Normal('T', data['T'][i], data['T_err'][i])
            M = Normal('M', data['M'][i], data['M_err'][i])
            X = Normal('X', data['X'][i], data['X_err'][i])
            if m_out_prior is not None:
                m_out_min = f_m_out_min(data['z'][i])
                m_out_max = f_m_out_max(data['z'][i])
                m_out = Uniform('m_out', float(m_out_min),
                                float(m_out_max))
            T_sigma = data['T_err'][i+1]
            M_sigma = data['M_err'][i+1]
            X_sigma = data['X_err'][i+1]
            cov = np.array([[T_sigma*T_sigma, 0., 0.],
                            [0., M_sigma*M_sigma, 0.],
                            [0., 0., X_sigma*X_sigma]])
            T_next = data['T'][i+1]
            M_next = data['M'][i+1]
            X_next = data['X'][i+1]

            y_next = np.array([T_next, M_next, X_next])
            dt = (dates[i+1] - dates[i])/pd.Timedelta('1D')
            ns = NestedSampling(seed=seed)
            _lh = LikeliHood(data=y_next, dt=dt, ws=ws, cov=cov,
                             date=dates[i], intmethod=intmethod)
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
                     {'dates': dates[:-1].values,
                      'dates_p': dates[1:].values,
                      'parameters': ['q_in', 'm_in', 'm_out',
                                     'h', 'T', 'M', 'X', 'dq_in'],
                      'obs': ['T', 'M', 'X', 'q_in', 'm_in',
                              'm_out', 'h', 'ws'],
                      'val_std': ['val', 'std']})
    res.to_netcdf(results_file)
    return res


def main(start, end, results_dir, pre_txt):

    ld = LakeData()
    data = ld.get_data(start, end)
    # Setup path for results file
    tstart = data.index[0]
    tend = data.index[-1]
    res_fn = 'ns_sampling_{:s}_{:s}.nc'
    res_fn = res_fn.format(tstart.strftime('%Y-%m-%d'),
                           tend.strftime('%Y-%m-%d'))
    if pre_txt is not None:
        res_fn = pre_txt + '_' + res_fn
    res_fn = os.path.join(results_dir, res_fn)
    
    ns_sampling(data, res_fn, nsamples=10000, nresample=500,
                q_in_min=0., q_in_max=1000., m_in_min=0., m_in_max=20.,
                m_out_min=0., m_out_max=20., new=False,
                m_out_prior=None, tolZ=1e-3, lh_fun=None,
                tolH=3., H=6., ws=4.5, seed=-1, intmethod='euler',
                gradient=False)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='mcmc_heat.py',
                            description=__doc__.strip())
    parser.add_argument('--rdir', type=str, default='./',
                        help='Directory to write results to.')
    parser.add_argument('--pretxt', type=str, default=None,
                        help='Text to prepend to file name.')
    parser.add_argument('-s', '--starttime', type=str,
                        default=datetime.utcnow()-timedelta(days=365),
                        help='Start of the data window')
    parser.add_argument('-e', '--endtime', type=str,
                        default=datetime.utcnow(),
                        help='End of the data window')
    args = parser.parse_args()
    main(start=args.starttime, end=args.endtime,
         results_dir=args.rdir, pre_txt=args.pretxt)




