"""
Solve the mass and energy balance model using
the Unscented Kalman Smoother.
"""
from functools import partial

from filterpy.kalman import (UnscentedKalmanFilter,
                             unscented_transform)

import numpy as np
import pandas as pd

from .syn_model import SynModel
from .forward_model import Forwardmodel
from .sigma_points import MerweScaledSigmaPoints


class Fx:

    def __init__(self, test=False):
        if test:
            s = SynModel()
            self.fm = Forwardmodel(method='rk4', mass2area=s.mass2area)
        else:
            self.fm = Forwardmodel(method='rk4')

    def run(self, x, dt, date):
        """
        Forward model lake temperature
        """
        y = np.zeros(8)
        y[0:8] = x[0:8]
        dp = [0.] * 5
        dp[0] = x[8]
        if len(x) == 13:
            dp[1] = x[9]
            dp[2] = x[10]
            dp[3] = x[11]
            dp[4] = x[12]
        y_next = self.fm.integrate(y, date, dt, dp)
        return np.r_[y_next[0:8], x[8:]]


def h_x(x):
    """
    Measurement function
    """
    return [x[0], x[1], x[2], x[5], x[7]]


class UnscentedKalmanSmoother:

    def __init__(self, data):
        self.data = data
        self.dt = (data.index[1] - data.index[0])/pd.Timedelta('1D')

    def residual_nan(self, x, y):
        """
        Compute the residual between x and y if one or both
        contain NaNs
        """
        rs = np.subtract(x, y)
        return rs[~np.isnan(rs)]
    
    def sigma_point_constraints(self, points):
        idx = np.where(points[:,:8] < 0.)
        points[idx] = 0.
        dpidx = (idx[0], idx[1] + 5)
        points[dpidx] = np.where(points[dpidx] < 0., 0., points[dpidx])
        return points


    def __call__(self, Q, X0, P0, test=False, alpha=1e-1, beta=2., kappa=0.):
        nvar = len(X0)
        points = MerweScaledSigmaPoints(n=nvar, alpha=alpha, beta=beta,
                                        kappa=kappa)
        F = Fx(test=test)
        kf = UnscentedKalmanFilter(dim_x=nvar, dim_z=5, dt=self.dt, fx=F.run,
                                   hx=h_x, points=points)
        kf.x = X0.values
        kf.Q = Q.values*self.dt*self.dt
        kf.P = P0.values
        nperiods = self.data.shape[0]-1
        Xs = np.zeros((nperiods, nvar))
        Ps = np.zeros((nperiods, nvar, nvar))
        kf.residual_z = self.residual_nan

        # Filtering
        log_lh = 0
        for i in range(nperiods):
            _fx = partial(F.run, date=self.data.index[i])
            kf.fx = _fx

            try:
                weights, sigmas = points.sigma_points(kf.x, kf.P)
                sigmas = self.sigma_point_constraints(sigmas)
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                raise e
            for k, s in enumerate(sigmas):
                try:
                    kf.sigmas_f[k] = kf.fx(s, self.dt)
                except ValueError as ve:
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
            z = self.data.iloc[i+1][['T', 'M', 'X', 'Mo', 'W']]
            T_err, M_err, X_err, Mo_err, W_err = self.data.iloc[i+1][['T_err',
                                                                      'M_err',
                                                                      'X_err',
                                                                      'Mo_err',
                                                                      'W_err']]
            kf.R = np.eye(5)*[T_err*T_err, M_err*M_err, X_err*X_err,
                              Mo_err*Mo_err, W_err*W_err]*self.dt**2
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
                Pxz = kf.cross_variance(kf.x, zp, kf.sigmas_f,
                                        kf.sigmas_h[:, z_mask])

                kf.K = np.dot(Pxz, kf.inv(Pz))        # Kalman gain
                kf.y = np.subtract(z[z_mask], zp)   # residual

                # update Gaussian state estimate (x, P)
                kf.x = kf.x + np.dot(kf.K, kf.y)
                kf.P = kf.P - np.dot(kf.K, np.dot(Pz, kf.K.T))

                kf._log_likelihood = None
                kf._likelihood = None
                # update likelihood
                try:
                    log_lh += kf.log_likelihood
                except:
                    import ipdb
                    ipdb.set_trace()
            Xs[i, :] = kf.x
            Ps[i, :, :] = kf.P

        # Smoothing
        xs, ps = Xs.copy(), Ps.copy()
        if True:
            n, dim_x = xs.shape
            num_sigmas = kf._num_sigmas
            dts = [kf._dt] * n
            sigmas_f = np.zeros((num_sigmas, dim_x))
            res = np.zeros((n, dim_x))

            for k in reversed(range(n-1)):
                # create sigma points from state estimate,
                # pass through state func
                weights, sigmas = points.sigma_points(xs[k], ps[k])
                sigmas = self.sigma_point_constraints(sigmas)
                for i in range(num_sigmas):
                    sigmas_f[i] = kf.fx(sigmas[i], dts[k],
                                        date=self.data.index[k])
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

        # Append smoothed values
        self.data['T_uks'] = np.r_[X0[0], xs[:, 0]]
        self.data['T_uks_err'] = np.r_[P0[0, 0], ps[:, 0, 0]]
        self.data['M_uks'] = np.r_[X0[1], xs[:, 1]]
        self.data['M_uks_err'] = np.r_[P0[1, 1], ps[:, 1, 1]]
        self.data['X_uks'] = np.r_[X0[2], xs[:, 2]]
        self.data['X_uks_err'] = np.r_[P0[2, 2], ps[:, 2, 2]]
        self.data['qi_uks'] = np.r_[X0[3]/0.0864, xs[:, 3]/0.0864]
        self.data['qi_uks_err'] = np.r_[P0[3, 3], ps[:, 3, 3]/0.0864]
        self.data['Mi_uks'] = np.r_[X0[4], xs[:, 4]]
        self.data['Mi_uks_err'] = np.r_[P0[4, 4], ps[:, 4, 4]]
        self.data['Mo_uks'] = np.r_[X0[5], xs[:, 5]]
        self.data['Mo_uks_err'] = np.r_[P0[5, 5], ps[:, 5, 5]]
        self.data['H_uks'] = np.r_[X0[6], xs[:, 6]]
        self.data['H_uks_err'] = np.r_[P0[6, 6], ps[:, 6, 6]]
        self.data['W_uks'] = np.r_[X0[7], xs[:, 7]]
        self.data['W_uks_err'] = np.r_[P0[7, 7], ps[:, 7, 7]]

        # Append filtered values
        self.data['T_ukf'] = np.r_[X0[0], Xs[:, 0]]
        self.data['T_ukf_err'] = np.r_[P0[0, 0], Ps[:, 0, 0]]
        self.data['M_ukf'] = np.r_[X0[1], Xs[:, 1]]
        self.data['M_ukf_err'] = np.r_[P0[1, 1], Ps[:, 1, 1]]
        self.data['X_ukf'] = np.r_[X0[2], Xs[:, 2]]
        self.data['X_ukf_err'] = np.r_[P0[2, 2], Ps[:, 2, 2]]
        self.data['qi_ukf'] = np.r_[X0[3]/0.0864, Xs[:, 3]/0.0864]
        self.data['qi_ukf_err'] = np.r_[P0[3, 3], Ps[:, 3, 3]/0.0864]
        self.data['Mi_ukf'] = np.r_[X0[4], Xs[:, 4]]
        self.data['Mi_ukf_err'] = np.r_[P0[4, 4], Ps[:, 4, 4]]
        self.data['Mo_ukf'] = np.r_[X0[5], Xs[:, 5]]
        self.data['Mo_ukf_err'] = np.r_[P0[5, 5], Ps[:, 5, 5]]
        self.data['H_ukf'] = np.r_[X0[6], Xs[:, 6]]
        self.data['H_ukf_err'] = np.r_[P0[6, 6], Ps[:, 6, 6]]
        self.data['W_ukf'] = np.r_[X0[7], Xs[:, 7]]
        self.data['W_ukf_err'] = np.r_[P0[7, 7], Ps[:, 7, 7]]
        return log_lh

    def likelihood(self, var, sid):
        T_Q = var[0]
        M_Q = var[1]
        X_Q = var[2]
        qi_Q = var[3]
        Mi_Q = var[4]
        Mo_Q = var[5]
        dqi_Q = var[6]
        dMi_Q = var[7]
        dMo_Q = var[8]

        qi0 = var[9]
        Mi0 = var[10]
        Mo0 = var[11]
        dqi0 = var[12]
        dMi0 = var[13]
        dMo0 = var[14]
        efact1 = var[15]
        efact2 = var[16]
        T0, M0, X0 = self.data.iloc[0][['T', 'M', 'X']]
        P0 = np.eye(9)*[T0*efact1, M0*efact1, X0*efact1,
                        qi0*efact2, Mi0*efact2, Mo0*efact2,
                        dqi0*efact2, dMi0*efact2, dMo0*efact2]
        Q = np.eye(9)*[T_Q, M_Q, X_Q, qi_Q, Mi_Q, Mo_Q,
                       dqi_Q, dMi_Q, dMo_Q]*self.dt**2
        X0 = [T0, M0, X0, qi0*0.0864, Mi0, Mo0,
              dqi0, dMi0, dMo0]
        try:
            log_lh = self.__call__(Q, X0, P0, test='True')
        except Exception as e:
            print(e)
            return 1e-10
        return log_lh
