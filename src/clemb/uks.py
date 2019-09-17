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
    return [x[0], x[1], x[2]]


class UnscentedKalmanSmoother:

    def __init__(self, data):
        self.data = data
        self.dt = (data.index[1] - data.index[0])/pd.Timedelta('1D')

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
                                             kf.Wm, kf.Wc, kf.Q,
                                             kf.x_mean, kf.residual_x)
            # save prior
            kf.x_prior = np.copy(kf.x)
            kf.P_prior = np.copy(kf.P)
            z = self.data.iloc[i+1][['T', 'M', 'X']]
            kf.update(z)
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
                for i in range(num_sigmas):
                    sigmas_h[i] = kf.hx(sigmas[i])

                # mean and covariance of prediction passed
                # through unscented transform
                zp, kf.S = unscented_transform(sigmas_h, kf.Wm, kf.Wc, kf.R)

                # recompute residual
                z = self.data.iloc[k][['T', 'M', 'X']].values
                kf.y = kf.residual_z(z, zp)
                Ks[k] = K
                # update likelihood
                log_lh += kf.log_likelihood
        return xs, ps, Xs, Ps, log_lh

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
        P0 = var[15]
        Q = np.eye(9)*[T_Q, M_Q, X_Q, qi_Q, Mi_Q, Mo_Q,
                       dqi_Q, dMi_Q, dMo_Q]*self.dt**2
        T0, M0, X0 = self.data.iloc[0][['T', 'M', 'X']]
        X0 = [T0, M0, X0, qi0*0.0864, Mi0, Mo0,
              dqi0, dMi0, dMo0]
        try:
            xs, ps, Xs, Ps, log_lh = self.__call__(Q, X0, P0, test='True')
        except Exception as e:
            print(e)
            return 1e-10
        return log_lh
