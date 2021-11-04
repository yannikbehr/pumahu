"""
Solve the mass and energy balance model using
the Unscented Kalman Smoother.
"""
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
import os

from filterpy.kalman import (UnscentedKalmanFilter,
                             unscented_transform)

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from . import get_data 
from .syn_model import SynModel
from .forward_model import Forwardmodel
from .sigma_points import MerweScaledSigmaPoints
from .data import LakeData
from .visualise import trellis_plot, plot_qin_uks
 
import ipdb


class Fx:

    def __init__(self, nparams, ngrads, test=False):
        """
        Run the forward model.

        Parameters:
        -----------
        nparams : int
                  Number of free parameters
        ngrads : int
                 Number of free gradients
        test: bool
              If 'True' use a mass2area function
              corresponding to the synthetic test setup.
        """
        self.nparams = nparams
        self.ngrads = ngrads
        if test:
            s = SynModel()
            self.fm = Forwardmodel(method='rk4', mass2area=s.mass2area)
        else:
            self.fm = Forwardmodel(method='rk4')

    def run(self, x, dt, date):
        """
        Forward model lake temperature
        """
        y = np.zeros(self.nparams)
        y[:] = x[0:self.nparams]
        dp = [0.] * self.ngrads 
        dp[:] = x[self.nparams:]
        y_next = self.fm.integrate(y, date, dt, dp)
        return np.r_[y_next[0:self.nparams], x[self.nparams:]]


def h_x(x):
    """
    Measurement function
    """
    return [x[0], x[1], x[2], x[5], x[6], x[7]]


class UnscentedKalmanSmoother:

    def __init__(self, data, X0=None, P0=None, Q=None,
                 initvals={'qi':200., 'm_in': 10, 'm_out':10., 'X':2.}):
        self.dates = pd.to_datetime(data['dates'].values)
        self.ndates = len(self.dates)
        self.dt = (self.dates[1] - self.dates[0])/pd.Timedelta('1D')
        self.params = ['T', 'M', 'X', 'q_in', 'm_in', 'm_out', 'h', 'W']
        self.nparams = len(self.params)
        try:
            self.data = data.loc[:, self.params, :]
            (T, M, X, Mi, Mo, qi, H, W) = self.data.loc[:,  self.params, 'val'].values[0]
        except KeyError:
            self.data = data.loc[:, ['T', 'M', 'X', 'm_out', 'h', 'W'], :]
            (T, M, X, Mo, H, W) = self.data.loc[:, ['T', 'M', 'X', 'm_out', 'h', 'W'], 'val'].values[0]
            qi, Mi, Mo, X = list(initvals.values())
            
        dqi = 1e-1
        dMi = 1e-1
        dMo = 1e-1
        dH = 1e-1
        dW = 1e-1
        self.ngrads = 5
        self.X0 = [T, M, X, qi*0.0864, Mi, Mo, H, W, dqi, dMi, dMo, dH, dW]
        if X0 is not None:
            self.X0 = X0
        self.nvar = len(self.X0)
                
        _P0 = OrderedDict(T=1e0, M=1e0, X=1e0, q_in=1e3,
                          m_in=1e3, m_out=1e3, h=1e-1, W=1e-1,
                          dqi=1e3, dMi=1e3, dMo=1e3, dH=1e-1, 
                          dW=1e-1)
        _P0 = np.eye(len(_P0))*list(_P0.values())
        self.P0 = _P0
        if P0 is not None:
            self.P0 = P0
            
        _Q = OrderedDict(T=1e-1, M=1e1, X=1e-3, q_in=1e2,
                         m_in=1e1, m_out=1e1, h=1e-3, W=1e-3,
                         dqi=0, dMi=0, dMo=0, dH=0, dW=0)
        _Q = np.eye(len(_Q))*list(_Q.values())
        self.Q = _Q
        if Q is not None:
            self.Q = Q
            
    def residual_nan(self, x, y):
        """
        Compute the residual between x and y if one or both
        contain NaNs
        """
        rs = np.subtract(x, y)
        return rs[~np.isnan(rs)]
    
    def sigma_point_constraints(self, points):
        """
        Set negative sigma points to zero.

        Parameters:
        -----------
        points : :class:`numpy.ndarray`
                 Sigma points
        Returns:
        --------
        :class:`numpy.ndarray`
            The constrained sigma points
        """
        # set negative sigma points for parameters to 0
        idx = np.where(points[:,:self.nparams] < 0.)
        points[idx] = 0.
        # now set the gradients to zero for which the
        # parameters were zero
        dpidx = (idx[0], idx[1] + self.ngrads)
        points[dpidx] = np.where(points[dpidx] < 0., 0., points[dpidx])
        return points

    def __call__(self, test=False, alpha=.3, beta=2., kappa=0.,
                 results_file=None, smooth=True):
        points = MerweScaledSigmaPoints(n=self.nvar, alpha=alpha, beta=beta,
                                        kappa=kappa)
        F = Fx(self.nparams, self.ngrads, test=test)
        kf = UnscentedKalmanFilter(dim_x=self.nvar, dim_z=5, dt=self.dt, fx=F.run,
                                   hx=h_x, points=points)
        kf.x = self.X0
        # The process noise should increase the larger
        # the time step but I haven't found any literature
        # on how to set the process noise properly
        kf.Q = self.Q
        kf.P = self.P0
        nperiods = self.data.shape[0]-1
        Xs = np.zeros((nperiods, self.nvar))
        Ps = np.zeros((nperiods, self.nvar, self.nvar))
        kf.residual_z = self.residual_nan
        p_parameters = ['lh']
        npparams = len(p_parameters)
        p_samples = np.zeros((self.ndates, npparams))*np.nan

        # Filtering
        for i in tqdm(range(nperiods)):
            _fx = partial(F.run, date=self.dates[i])
            dt = (self.dates[i+1] - self.dates[i])/pd.Timedelta('1D')
            kf.fx = _fx

            try:
                weights, sigmas = points.sigma_points(kf.x, kf.P)
                sigmas = self.sigma_point_constraints(sigmas)
            except Exception as e:
                print(e)
                raise e

            for k, s in enumerate(sigmas):
                try:
                    kf.sigmas_f[k] = kf.fx(s, dt)
                except ValueError as ve:
                    print(ve)
                    print(s)
                    print(kf.sigmas_f)
                    print(kf.fx)
                    raise ve
            kf.x, kf.P = unscented_transform(kf.sigmas_f,
                                             weights[0],
                                             weights[1], kf.Q)
            # save prior
            kf.x_prior = np.copy(kf.x)
            kf.P_prior = np.copy(kf.P)
            dtpoint = self.data.isel(dates=i+1)
            z = dtpoint.loc[['T', 'M', 'X', 'm_out', 'h', 'W'], 'val'].values
            z_err = dtpoint.loc[['T', 'M', 'X', 'm_out', 'h', 'W'], 'std'].values
            kf.R = np.eye(z_err.size)*(z_err*z_err)
            # if measurements or measurement error are all
            # NaN, don't update
            if not np.alltrue(np.isnan(z)) and not np.alltrue(np.isnan(kf.R)):

                # Handle NaNs
                z_mask = ~np.isnan(z)
                R_mask = np.outer(z_mask, z_mask)
                rmdim = np.sum(z_mask)

                sigmas_h = []
                for s in kf.sigmas_f:
                    sigmas_h.append(kf.hx(s))

                kf.sigmas_h = np.atleast_2d(sigmas_h)

                # mean and covariance of prediction passed through
                # unscented transform
                zp, Pz = unscented_transform(kf.sigmas_h[:, z_mask],
                                             weights[0], weights[1],
                                             kf.R[R_mask].reshape(rmdim,
                                                                  rmdim))

                kf.S = Pz
                # compute cross variance of the state and the measurements
                try:
                    Pxz = kf.cross_variance(kf.x, zp, kf.sigmas_f,
                                            kf.sigmas_h[:, z_mask])
                except Exception as e:
                    print(e)

                kf.K = np.dot(Pxz, kf.inv(Pz))        # Kalman gain
                kf.y = np.subtract(z[z_mask], zp)   # residual

                # update Gaussian state estimate (x, P)
                kf.x = kf.x + np.dot(kf.K, kf.y)
                kf.P = kf.P - np.dot(kf.K, np.dot(Pz, kf.K.T))

                kf._log_likelihood = None
                kf._likelihood = None
                # update likelihood
                p_samples[i+1, 0] = kf.log_likelihood

            Xs[i, :] = kf.x
            Ps[i, :, :] = kf.P

        # Smoothing
        xs, ps = Xs.copy(), Ps.copy()
        if smooth:
            n, dim_x = xs.shape
            num_sigmas = kf._num_sigmas
            dts = [kf._dt] * n
            sigmas_f = np.zeros((num_sigmas, dim_x))
            res = np.zeros((n, dim_x))

            for k in tqdm(reversed(range(n-1))):
                # ipdb.set_trace(cond=(self.dates[k] < pd.Timestamp(2020, 4, 28)))
                # create sigma points from state estimate,
                # pass through state func
                weights, sigmas = points.sigma_points(xs[k], ps[k])
                sigmas = self.sigma_point_constraints(sigmas)
                for i in range(num_sigmas):
                    sigmas_f[i] = kf.fx(sigmas[i], dts[k],
                                        date=self.dates[k])
                xb, Pb = unscented_transform(sigmas_f, weights[0],
                                             weights[1], kf.Q)

                # compute cross variance
                Pxb = 0
                for i in range(num_sigmas):
                    y = kf.residual_x(sigmas_f[i], xb)
                    z = kf.residual_x(sigmas[i], xs[k])
                    Pxb += kf.Wc[i] * np.outer(z, y)

                # compute gain
                K = np.dot(Pxb, kf.inv(Pb))

                # update the smoothed estimates
                xs[k] += np.dot(K, kf.residual_x(xs[k+1], xb))
                res[k] = kf.residual_x(xs[k+1], xb)
                ps[k] += np.dot(K, ps[k+1] - Pb).dot(K.T)

        exp = np.zeros((self.ndates, self.nparams, 2))*np.nan
        T_idx = self.params.index('T')
        exp[:, T_idx, 0] = np.r_[self.X0[T_idx], xs[:, T_idx]]
        exp[:, T_idx, 1] = np.r_[self.P0[T_idx, T_idx], ps[:, T_idx, T_idx]]
        M_idx = self.params.index('M')
        exp[:, M_idx, 0] = np.r_[self.X0[M_idx], xs[:, M_idx]]
        exp[:, M_idx, 1] = np.r_[self.P0[M_idx, M_idx], ps[:, M_idx, M_idx]]
        X_idx = self.params.index('X')
        exp[:, X_idx, 0] = np.r_[self.X0[X_idx], xs[:, X_idx]]
        exp[:, X_idx, 1] = np.r_[self.P0[X_idx, X_idx], ps[:, X_idx, X_idx]]
        q_in_idx = self.params.index('q_in')
        exp[:, q_in_idx, 0] = np.r_[self.X0[q_in_idx] / 0.0864,
                                    xs[:, q_in_idx] / 0.0864]
        exp[:, q_in_idx, 1] = np.r_[self.P0[q_in_idx, q_in_idx] / 0.0864,
                                    ps[:, q_in_idx, q_in_idx] / 0.0864]
        m_in_idx = self.params.index('m_in')
        exp[:, m_in_idx, 0] = np.r_[self.X0[m_in_idx], xs[:, m_in_idx]]
        exp[:, m_in_idx, 1] = np.r_[self.P0[m_in_idx, m_in_idx],
                                    ps[:, m_in_idx, m_in_idx]]
        m_out_idx = self.params.index('m_out')
        exp[:, m_out_idx, 0] = np.r_[self.X0[m_out_idx], xs[:, m_out_idx]]
        exp[:, m_out_idx, 1] = np.r_[self.P0[m_out_idx, m_out_idx],
                                     ps[:, m_out_idx, m_out_idx]]
        h_idx = self.params.index('h')
        exp[:, h_idx, 0] = np.r_[self.X0[h_idx], xs[:, h_idx]]
        exp[:, h_idx, 1] = np.r_[self.P0[h_idx, h_idx], ps[:, h_idx, h_idx]]
        W_idx = self.params.index('W')
        exp[:, W_idx, 0] = np.r_[self.X0[W_idx], xs[:, W_idx]]
        exp[:, W_idx, 1] = np.r_[self.P0[W_idx, W_idx], ps[:, W_idx, W_idx]]
        res = xr.Dataset({'exp': (('dates', 'parameters', 'val_std'), exp),
                          'p_samples': (('dates', 'p_parameters'), p_samples),
                          'input': (('dates', 'i_parameters', 'val_std'),
                                    self.data.data)},
                         {'dates': self.dates,
                          'parameters': self.params,
                          'p_parameters': p_parameters,
                          'i_parameters': self.data['parameters'].values,
                          'val_std': ['val', 'std']})
        
        if results_file is not None:
            res.to_netcdf(results_file)
        return res

    def likelihood(self, var):
        P0 = OrderedDict(T=1e0, M=1e0, X=1e0, q_in=1e1,
                         m_in=1e3, m_out=1e3, h=1e-1, W=1e-1,
                         dqi=1e-1, dMi=1e-1, dMo=1e-1, dH=1e-1, 
                         dW=1e-1)
        P0 = np.eye(len(P0))*list(P0.values())

        Q = OrderedDict(T=var[0], M=var[1], X=var[2], q_in=var[3],
                        m_in=var[4], m_out=var[5], h=var[6], W=var[7],
                        dqi=var[8], dMi=var[9], dMo=var[10], dH=var[11],
                        dW=var[12])
        Q = np.eye(len(Q))*list(Q.values())
        self.Q = Q
        self.P0 = P0
        try:
            log_lh = self.__call__()
        except Exception as e:
            print(e)
            return 1e-10
        return log_lh

    
def mainCore(args):
    
    # If the script runs in daemon mode update the start
    # and end time
    if args.daemon:
        args.starttime = datetime.utcnow()-timedelta(days=365)
        args.endtime = datetime.utcnow()
    # Setup path for results file
    res_fn = 'uks.nc'
    if args.pretxt is not None:
        res_fn = args.pretxt + '_' + res_fn
    res_fn = os.path.join(args.rdir, res_fn)
    
    if args.fit:
        ld = LakeData()
        data = ld.get_data(args.starttime, args.endtime, smoothing='dv')
        uks = UnscentedKalmanSmoother(data=data)
        xds_uks = uks(results_file=res_fn)
    if args.plot:
        xdf = xr.open_dataset(res_fn)
        fout_trellis = os.path.join(args.rdir, 'uks_trellis.png')
        if args.benchmark is not None:
            data2 = xr.open_dataset(args.benchmark)
            trellis_plot(xdf, data2=data2, filename=fout_trellis)
            plot_qin_uks(xdf, annotations=True,
                         filename=os.path.join(args.rdir, 'uks_exp.png'))
        else:
            trellis_plot(xdf, filename=fout_trellis)
            plot_qin_uks(xdf, annotations=True,
                         filename=os.path.join(args.rdir, 'uks_exp.png'))


def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='heat_uks',
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
                        help='Run the Unscented Kalman smoother.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the results.')
    parser.add_argument('-d', '--daemon', action='store_true',
                        help='Run the script in daemon mode.')
    parser.add_argument('--benchmark', type=str,
                        default=None,
                        help='File containing benchmark solution.')
    args = parser.parse_args(argv)
    mainCore(args)

    
if __name__ == '__main__':
    main()
