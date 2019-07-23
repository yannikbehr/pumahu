import numpy as np
import pandas as pd
from scipy.signal import tukey
from scipy.stats import gamma
from clemb.forward_model import Forwardmodel


class SynModel:
    """
    Compute synthetic observations

    Compute synthetic observations by assuming a time series of
    input parameters for the mass and energy balance model. We further
    assume that the lake has a cylindrical shape with a give area and
    an outflow at 45.31 m. The outflow rate is computed using Bernoulli's
    equation.
    """

    def __init__(self, area=194162, T0=15.):
        self.f = 1/15.
        self.tmax = 30.
        self.a = area
        self.T0 = T0

    def outflow(self, level, area=0.2):
        """
        Compute outflow based on bernoulli equation.
        Returns outflow in km^3/day.
        """
        g = 9.81  # m/s^2
        h = level - 45.31
        if h <= 0.:
            return 0.
        v = np.sqrt(2*g*h)
        vol = area * v  # ouflow volume in m^3/s
        vol *= 1e-3 * 86400  # convert to km^3/day
        return vol

    def volume(self, mass, temp):
        """
        Compute volume based on mass and temperature.
        Return volume in km^3
        """
        rho = 1.003 - 0.00033 * temp
        return mass/rho

    def mass(self, volume, temp):
        """
        Return mass based on volume and temperature.
        """
        rho = 1.003 - 0.00033 * temp
        return volume * rho

    def level(self, volume, lakearea):
        """
        Compute lake level based on volume and surface area.
        Return lake level in m.
        """
        return volume * 1e3 / lakearea

    def synth_fullness(self, level):
        vol = self.a * level
        return self.a, vol/1e3

    def mass2area(self, mass, temp):
        """
        Compute the lake surface area from the mass and temperature.
        """
        v = self.volume(mass, temp)
        return self.a, v

    def run(self, q_in, mode='gamma', nsteps=100, gradient=False,
            integration_method='euler'):
        """
        Produce synthetic observations.
        """
        dates = pd.date_range(start='1/1/2017', end='21/1/2017',
                              periods=nsteps)
        t = np.linspace(0, self.tmax, nsteps)
        dt = (dates[1] - dates[0])/pd.Timedelta('1D')
        if mode == 'sinus':
            qi = np.sin(2.*np.pi*self.f*t + np.deg2rad(15))*q_in/2.+q_in/2.
            qi *= tukey(nsteps, 0.9)
        if mode == 'tukey':
            tail = int(10/dt)
            qi = np.ones(nsteps-tail)*q_in
            qi *= tukey(nsteps-tail, 0.9)
            qi_tmp = np.zeros(nsteps)
            qi_tmp[0:nsteps-tail] = qi
            qi = qi_tmp
        if mode == 'step':
            tail = int(10/dt)
            qi = np.ones(nsteps-tail)*q_in
            qi_tmp = np.zeros(nsteps)
            qi_tmp[0:nsteps-tail] = qi
            qi = qi_tmp
        if mode == 'gamma':
            y = gamma.pdf(t, 3., loc=4)
            y /= y.max()
            y *= q_in
            qi = y

        if mode == 'observed':
            # a polynomial model of a real inversion
            z = np.array([-6.93795125e-30,  1.17673746e-26, -9.19940930e-24,
                          4.39229128e-21, -1.43037875e-18,  3.36004438e-16,
                          -5.87098984e-14,  7.75486512e-12, -7.79213224e-10,
                          5.94657977e-08, -3.41657063e-06,  1.45392771e-04,
                          -4.46901292e-03,  9.56540064e-02, -1.35189993e+00,
                          1.16488040e+01, -5.37668791e+01,  1.05017651e+02,
                          2.91772946e+01])
            p = np.poly1d(z)
            dates = pd.date_range(start='2018-4-1', end='2018-9-30',
                                  periods=183)
            t = np.r_[0, np.cumsum(np.diff(dates)/np.timedelta64(1, 'D'))]
            qi = p(t)
            nsteps = 183

        if mode == 'test':
            qi = np.array([0., .2, .6, 1.])*q_in
            dates = pd.date_range(start='2017-01-01', end='2017-01-04',
                                  freq='D')
            nsteps = 4

        y = np.zeros((nsteps, 8))
        prm = np.zeros((nsteps, 2))
        V = 8800
        A = self.a
        Mi = 10.
        H = 6.0
        ws = 4.5
        X = 2.
        M = self.mass(V, self.T0)
        ll = self.level(V, A)
        Mo = self.outflow(ll)
        y[0, :] = [self.T0, M, X, qi[0]*0.0864,
                   Mi, Mo, H, ws]
        fm = Forwardmodel(method=integration_method,
                          mass2area=self.mass2area)
        for i in range(nsteps-1):
            prm[i, 0] = ll
            dt = (dates[i+1] - dates[i])/pd.Timedelta('1D')
            if not gradient:
                dp = [0.] * 5
                y[i, 3] = qi[i]*0.0864
                y[i, 5] = Mo
            else:
                dp = [0.] * 5
                dp[0] = (qi[i+1] - qi[i])*0.0864/dt
                y[i, 5] = Mo
            y_new = fm.integrate(y[i], dates[i], dt, dp)
            prm[i, 1] = fm.get_evap()[1]
            V = self.volume(y_new[1], y_new[0])
            ll = self.level(V, A)
            Mo = self.outflow(ll)
            y[i+1, :] = y_new.copy()
        prm[i+1, 0] = ll
        prm[i+1, 1] = fm.get_evap()[1]

        # Prescripe errors
        factor = 1.
        T_err = factor*0.4*dt
        z_err = factor*0.01*dt
        M_err = factor*2.*dt
        X_err = factor*0.4*dt
        A_err = factor*30.*dt
        V_err = factor*2.0*dt
        Mg_err = factor*50*dt

        # Design dataset
        syn_data = {}
        syn_data['T'] = y[:, 0]
        syn_data['T_err'] = np.ones(nsteps)*T_err
        syn_data['p_mean'] = 1.003 - 0.00033 * syn_data['T']
        syn_data['p_err'] = 0.00033*np.random.normal(scale=T_err, size=nsteps)
        syn_data['a'] = y[:, 6]
        syn_data['a_err'] = np.ones(nsteps)*A_err
        syn_data['M'] = y[:, 1]
        syn_data['M_err'] = np.ones(nsteps)*M_err
        syn_data['v'] = self.volume(syn_data['M'], syn_data['T'])
        syn_data['v_err'] = V_err
        syn_data['X'] = y[:, 2]
        syn_data['X_err'] = np.ones(nsteps)*X_err
        syn_data['Mg'] = syn_data['X']/syn_data['M']*1e6
        syn_data['Mg_err'] = Mg_err
        syn_data['z'] = prm[:, 0]
        syn_data['z_err'] = np.ones(nsteps)*z_err
        syn_data['W'] = np.ones(nsteps)*4.5
        syn_data['H'] = np.ones(nsteps)*6.0
        syn_data['dv'] = np.ones(nsteps)*1.0
        syn_data['Mi'] = y[:, 4]
        syn_data['Mo'] = y[:, 5]
        syn_data['mevap'] = prm[:, 1]
        syn_data['qi'] = y[:, 3]/0.0864
        return pd.DataFrame(syn_data, index=dates)
