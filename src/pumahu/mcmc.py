"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import xarray as xr

from nsampling import (NestedSampling, Uniform,
                       Normal, InvCDF, Constant)

from .forward_model import Forwardmodel
from .data import LakeData
from .visualise import (trellis_plot,
                        mcmc_heat_input,
                        heat_vs_rsam)


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
            self.samples.append(np.r_[y_new,
                                      np.array([self.fm.get_evap()[1],
                                                lh, sid])])
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
    if not new and os.path.isfile(results_file):
        res = xr.open_dataset(results_file)
        res.close()
        return res
    
    data = data.loc[:, ['T', 'M', 'X', 'z'], :]
    dates = pd.to_datetime(data['dates'].values)
    # setup model parameters
    ndates = dates.size
    # inversion parameters
    parameters = ['q_in', 'm_in', 'm_out',
                  'h', 'T', 'M', 'X', 'dq_in']
    nparams = len(parameters)
    nsvars = OrderedDict({param: None for param in parameters})
    # sampling performance parameters
    p_parameters = ['lh', 'wt', 'zs', 'hs', 'ig', 'Z', 'Z_var']
    npparams = len(p_parameters)
    
    nsvars['q_in'] = Uniform('qin', q_in_min, q_in_max)
    nsvars['m_in'] = Uniform('m_in', m_in_min, m_in_max)
    nsvars['h'] = Constant('h', H)
    f_m_out_min = None
    f_m_out_max = None
    if m_out_prior is not None:
        a = np.load(m_out_prior)
        z, m_out_min, m_out_max = a['z'], a['o_min'], a['o_max']
        f_m_out_min = interp1d(z, m_out_min, fill_value='extrapolate')
        f_m_out_max = interp1d(z, m_out_max, fill_value='extrapolate')
    else:
        nsvars['m_out'] = Uniform('m_out', m_out_min, m_out_max)

    if gradient:
        nsvars['dq_in'] = Uniform('dq', -2e3, 2e3)
    else:
        nsvars['dq_in'] = Constant('dq', 0.)

    if nresample < 0:
        nrsp = nsamples
    else:
        nrsp = nresample

    # return values
    ns_samples = np.zeros((ndates, nrsp, nparams))*np.nan
    p_samples = np.zeros((ndates, nrsp, npparams))*np.nan
    exp = np.zeros((ndates, nparams, 2))*np.nan
    MAP = np.zeros((ndates, nparams+1))*np.nan
    ig = np.zeros((ndates))*np.nan
    z = np.zeros((ndates, 2))*np.nan
    model_data = np.zeros((ndates, nrsp, nparams))*np.nan

    cov = np.zeros((3,3))
    y_next = np.zeros(3)*np.nan

    for i in tqdm(range(ndates-1)):
        # Take samples from the input
        for j, _v in enumerate(['T', 'M', 'X']):
            val_now = data[i].sel(i_parameters=_v, val_std='val')
            err_now = data[i].sel(i_parameters=_v, val_std='std')
            val_next = data[i+1].sel(i_parameters=_v, val_std='val')
            err_next = data[i+1].sel(i_parameters=_v, val_std='std')
            nsvars[_v] = Normal(_v, val_now, err_now)
            y_next[j] = val_next
            cov[j,j] = err_next*err_next
            
        if m_out_prior is not None:
            lake_level = data[i].sel(i_parameters='z', val_std='val')
            m_out_min = f_m_out_min(lake_level)
            m_out_max = f_m_out_max(lake_level)
            nsvars['m_out'] = Uniform('m_out', float(m_out_min),
                                      float(m_out_max))
        dt = (dates[i+1] - dates[i])/pd.Timedelta('1D')
        ns = NestedSampling(seed=seed)
        if lh_fun is None:
            _lh = LikeliHood(data=y_next, dt=dt, ws=ws, cov=cov,
                             date=dates[i], intmethod=intmethod)
            _lh_fun = _lh.run_lh
        else:
            _lh_fun = lh_fun
        rs = ns.explore(list(nsvars.values()), 100, nsamples, _lh_fun, 40,
                        0.1, tolZ, tolH)

        if nresample > 0:
            smp = rs.resample_posterior(nresample)
        else:
            smp = rs.get_samples()

        # store results
        exp[i, :, 0] = rs.getexpt()
        exp[i, :, 1] = rs.getvar()
        MAP[i, :] = rs.getmax()
        p_samples[i, :, 4] = rs.getH()
        Z, Z_var = rs.getZ()
        p_samples[i, :, 5] = Z
        p_samples[i, :, 6] = Z_var
        lh_samples = _lh.get_samples()
        for j, _s in enumerate(smp):
            sid = np.where(lh_samples[:, -1] == _s.get_id())
            y_mod = lh_samples[sid][0][:nparams]
            model_data[i+1, j, :] = y_mod[:]
            ns_samples[i, j, :] = _s.get_value() 
            p_samples[i, j, 0] = np.exp(_s.get_logL())
            p_samples[i, j, 1] = np.exp(_s.get_logWt())
            p_samples[i, j, 2] = _s.get_logZ()
            p_samples[i, j, 3] = _s.get_H()

        del smp, ns, rs

    # Save results to disk
    res = xr.Dataset({'exp': (('dates', 'parameters', 'val_std'), exp),
                      'MAP': (('dates', 'parameters'), MAP[:, :-1]),
                      'ns_samples': (('dates', 'samples', 'parameters'),
                                     ns_samples),
                      'p_samples': (('dates', 'samples', 'p_parameters'),
                                     p_samples),
                      'model': (('dates', 'samples', 'obs'), model_data),
                      'input': (('dates', 'i_parameters', 'val_std'),
                                data)},
                     {'dates': dates.values,
                      'parameters': parameters,
                      'p_parameters': p_parameters, 
                      'i_parameters': data['i_parameters'],
                      'obs': ['T', 'M', 'X', 'q_in', 'm_in',
                              'm_out', 'h', 'ws'],
                      'val_std': ['val', 'std']})

    res.to_netcdf(results_file)
    return res


def main(argv=None):
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
    parser.add_argument('-f', '--fit', action='store_true',
                        help='Run the MCMC sampling.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the results.')
    parser.add_argument('--prior', type=str, default=None,
                        help='File containing priors.')
    args = parser.parse_args(argv)
    ld = LakeData()
    data = ld.get_data(args.starttime, args.endtime)
    # Setup path for results file
    tstart = pd.to_datetime(data['dates'].values[0])
    tend = pd.to_datetime(data['dates'].values[-1])
    res_fn = 'mcmc_sampling_{:s}_{:s}.nc'
    res_fn = res_fn.format(tstart.strftime('%Y-%m-%d'),
                           tend.strftime('%Y-%m-%d'))
    if args.pretxt is not None:
        res_fn = args.pretxt + '_' + res_fn
    res_fn = os.path.join(args.rdir, res_fn)
    
    if args.fit:
        ns_sampling(data, res_fn, nsamples=1001, nresample=500,
                    q_in_min=0., q_in_max=1000., m_in_min=0., m_in_max=20.,
                    m_out_min=0., m_out_max=20., new=True,
                    m_out_prior=args.prior, tolZ=1e-3, lh_fun=None,
                    tolH=3., H=6., ws=4.5, seed=-1, intmethod='euler',
                    gradient=False)
    if args.plot:
        xdf = xr.open_dataset(res_fn)
        fout_trellis = os.path.join(args.rdir, 'mcmc_trellis.png')
        trellis_plot(xdf, filename=fout_trellis)
        fout_heat_input = os.path.join(args.rdir, 'mcmc_heat_input.png')
        mcmc_heat_input(xdf, filename=fout_heat_input)
        fout_heat_rsam = os.path.join(args.rdir, 'mcmc_heat_vs_rsam.png')
        heat_vs_rsam(xdf, filename=fout_heat_rsam)


if __name__ == '__main__':
   main()
