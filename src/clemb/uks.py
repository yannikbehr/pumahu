"""
Solve the mass and energy balance model using
the Unscented Kalman Smoother.
"""
from functools import partial

from filterpy.kalman import (UnscentedKalmanFilter,
                             MerweScaledSigmaPoints,
                             unscented_transform)

import numpy as np
import pandas as pd

from clemb.syn_model import SynModel
from clemb.forward_model import Forwardmodel


class Fx:

    def __init__(self, test=False):
        if test:
            s = SynModel()
            self.fm = Forwardmodel(method='rk4', mass2area=s.mass2area)
        else:
            self.fm = Forwardmodel(method='rk4')

    def run(self, x, dt, date, h=6.0, ws=4.5):
        """
        Forward model lake temperature
        """
        y = np.zeros(8)
        y[0:6] = x[0:6]
        y[6] = h
        y[7] = ws
        dp = [0.] * 5
        dp[0] = x[6]
        if len(x) == 9:
            dp[1] = x[7]
            dp[2] = x[8]
        y_next = self.fm.integrate(y, date, dt, dp)
        return np.r_[y_next[0:6], x[6:]]


def h_x(x):
    """
    Measurement function
    """
    # if observations contain NaNs, only return
    # those values with no NaNs
    return [x[0], x[1], x[2]]


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

    def __call__(self, Q, X0, P0, test=False):
        nvar = len(X0)
        points = MerweScaledSigmaPoints(n=nvar, alpha=1e-1, beta=2., kappa=0.)
        F = Fx(test=test)
        kf = UnscentedKalmanFilter(dim_x=nvar, dim_z=3, dt=self.dt, fx=F.run,
                                   hx=h_x, points=points)
        T_err, M_err, X_err = self.data.iloc[0][['T_err', 'M_err', 'X_err']]
        kf.x = X0
        kf.Q = Q
        kf.P = P0
        kf.R = np.eye(3)*[T_err*T_err, M_err*M_err, X_err*X_err]*self.dt**2
        nperiods = self.data.shape[0]-1
        Xs = np.zeros((nperiods, nvar))
        Ps = np.zeros((nperiods, nvar, nvar))
        kf.residual_z = self.residual_nan

        # Filtering
        for i in range(nperiods):
            _fx = partial(F.run, h=6.0, ws=4.5, date=self.data.index[i])
            kf.fx = _fx

            try:
                sigmas = kf.points_fn.sigma_points(kf.x, kf.P)
            except Exception as e:
                raise e
            if True:
                sigmas[:, 0:6] = np.where(sigmas[:, 0:6] < 0., 0.,
                                          sigmas[:, 0:6])
            for k, s in enumerate(sigmas):
                try:
                    kf.sigmas_f[k] = kf.fx(s, self.dt)
                except ValueError as ve:
                    print(s)
                    print(kf.sigmas_f)
                    print(kf.fx)
                    raise ve
            kf.x, kf.P = unscented_transform(kf.sigmas_f,
                                             kf.Wm, kf.Wc, kf.Q)
            # save prior
            kf.x_prior = np.copy(kf.x)
            kf.P_prior = np.copy(kf.P)
            z = self.data.iloc[i+1][['T', 'M', 'X']]
            T_err, M_err, X_err = self.data.iloc[i+1][['T_err',
                                                       'M_err',
                                                       'X_err']]
            kf.R = np.eye(3)*[T_err*T_err, M_err*M_err, X_err*X_err]*self.dt**2
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
                zp, Pz = unscented_transform(kf.sigmas_h[:, z_mask], kf.Wm, kf.Wc,
                                             kf.R[R_mask].reshape(rmdim, rmdim))

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

            Xs[i, :] = kf.x
            Ps[i, :, :] = kf.P

        # Smoothing
        xs, ps = Xs.copy(), Ps.copy()
        log_lh = 0
        if True:
            n, dim_x = xs.shape
            dim_z = 3
            num_sigmas = kf._num_sigmas
            dts = [kf._dt] * n
            sigmas_f = np.zeros((num_sigmas, dim_x))
            Ks = np.zeros((n, dim_x, dim_x))
            res = np.zeros((n, dim_x))

            for k in reversed(range(n-1)):
                # create sigma points from state estimate,
                # pass through state func
                sigmas = kf.points_fn.sigma_points(xs[k], ps[k])
                for i in range(num_sigmas):
                    sigmas_f[i] = kf.fx(sigmas[i], dts[k],
                                        date=self.data.index[k])
                xb, Pb = unscented_transform(sigmas_f, kf.Wm, kf.Wc, kf.Q)

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

                # recompute S based on the smoothed state values
                sigmas_h = np.zeros((num_sigmas, dim_z))
                sigmas = kf.points_fn.sigma_points(xs[k], ps[k])
                z = self.data.iloc[k][['T', 'M', 'X']].values
                T_err, M_err, X_err = self.data.iloc[k][['T_err',
                                                         'M_err',
                                                         'X_err']]
                kf.R = np.eye(3)*[T_err*T_err,
                                  M_err*M_err,
                                  X_err*X_err]*self.dt**2

                # Handle NaNs
                z_mask = ~np.isnan(z)
                R_mask = np.outer(z_mask, z_mask)
                rmdim = np.sum(z_mask)

                for i in range(num_sigmas):
                    sigmas_h[i] = kf.hx(sigmas[i])

                # mean and covariance of prediction passed
                # through unscented transform
                R_masked = kf.R[R_mask].reshape(rmdim, rmdim)
                zp, kf.S = unscented_transform(sigmas_h[:, z_mask],
                                               kf.Wm, kf.Wc, R_masked)

                # recompute residual
                kf.y = kf.residual_z(z[z_mask], zp)
                Ks[k] = K
                # update likelihood
                log_lh += kf.log_likelihood

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
        P0 = np.eye(9)*var[15]
        Q = np.eye(9)*[T_Q, M_Q, X_Q, qi_Q, Mi_Q, Mo_Q,
                       dqi_Q, dMi_Q, dMo_Q]*self.dt**2
        T0, M0, X0 = self.data.iloc[0][['T', 'M', 'X']]
        X0 = [T0, M0, X0, qi0*0.0864, Mi0, Mo0,
              dqi0, dMi0, dMo0]
        try:
            log_lh = self.__call__(Q, X0, P0, test='True')
        except Exception as e:
            print(e)
            return 1e-10
        return log_lh
