from datetime import date, timedelta

from cachier import cachier
import numpy as np
import pandas as pd
from scipy.signal import tukey
from scipy.stats import gamma
import xarray as xr
from tqdm import tqdm

from .forward_model import Forwardmodel


class SynModel:
    """
    Compute synthetic observations

    Compute synthetic observations by assuming a time series of
    input parameters for the mass and energy balance model. We further
    assume that the lake has a cylindrical shape with a given area and
    an outflow at 45.31 m. The outflow rate is computed using Bernoulli's
    equation.
    """

    def __init__(self, area=194162, T0=15., integration_method='euler',
                 seed=None):
        self.f = 1/15.
        self.tmax = 30.
        self.a = area
        self.T0 = T0
        if seed is not None:
            np.random.seed(seed)
        self.fm = Forwardmodel(method=integration_method,
                               mass2area=self.mass2area)

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
            addnoise=False, 
            estimatenoise=False):
        """
        Produce synthetic observations.
        """
        np.random.seed(42)
        dates = pd.date_range(start='2017-01-01', end='2017-01-21',
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
        if mode == 'constant':
            qi = np.ones(nsteps)*q_in
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
            
        if mode == 'gamma+sin':
            y = gamma.pdf(t, 3., loc=4)
            y /= y.max()
            y *= q_in
            npts = y.size
            t1 = t[npts//2::]
            t1 = np.arange(t1.size)
            y[npts//2:] = np.sin(2.*np.pi*self.f*t1 + 2*np.pi)*q_in/4.+q_in/4.
            y[npts//2:] *= tukey(t1.size, 0.9)
            qi = y
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
            y_new = self.fm.integrate(y[i], dates[i], dt, dp)
            prm[i, 1] = self.fm.get_evap()[1]
            V = self.volume(y_new[1], y_new[0])
            ll = self.level(V, A)
            Mo = self.outflow(ll)
            y[i+1, :] = y_new.copy()
        prm[i+1, 0] = ll
        prm[i+1, 1] = self.fm.get_evap()[1]

        # Prescripe errors
        factor = 1.
        T_err = factor*0.4*dt
        z_err = factor*0.01*dt
        M_err = factor*2.*dt
        X_err = factor*0.4*dt
        A_err = factor*30.*dt
        V_err = factor*2.0*dt
        Mg_err = factor*50*dt
        Mo_err = factor*.25
        W_err = factor*.5

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
        syn_data['W_err'] = np.ones(nsteps)*W_err
        syn_data['H'] = np.ones(nsteps)*6.0
        syn_data['dv'] = np.ones(nsteps)*1.0
        syn_data['Mi'] = y[:, 4]
        syn_data['Mo'] = y[:, 5]
        syn_data['Mo_err'] = Mo_err*y[:, 5]
        syn_data['mevap'] = prm[:, 1]
        syn_data['qi'] = y[:, 3]/0.0864

        if addnoise:
            syn_data['T'] += np.random.normal(scale=T_err, size=nsteps)
            syn_data['M'] += np.random.normal(scale=M_err, size=nsteps)
            syn_data['X'] += np.random.normal(scale=X_err, size=nsteps)

        df = pd.DataFrame(syn_data, index=dates)
        if estimatenoise:
            df_mean = df.groupby(pd.Grouper(freq='D')).mean()
            df_std = df.groupby(pd.Grouper(freq='D')).std()
            df_mean['T_err'] = df_std['T']
            df_mean['M_err'] = df_std['M']
            df_mean['X_err'] = df_std['X']
            df = df_mean

        xds = xr.Dataset(df)
        xds = xds.rename({'dim_0': 'dates'})
        return xds


def lhs(dim, nsamples, centered=False):
    """
    Compute latin hypercube samples.
    
    :param dim: Dimensionality of the samples
    :type dim: int
    :param nsamples: Number of samples
    :type nsamples: int
    :param centered: If 'True' returns samples
                     centered within intervals
    :type centered: bool
    :returns: Latin hypercube samples with dimension
              nsamples x dim
    :rtype: :class:`numpy.ndarray`
    """
    # Make the random pairings
    H = np.zeros((nsamples, dim))
    if centered:
        # Generate the intervals
        cut = np.linspace(0, 1, nsamples + 1)
        # Fill points uniformly in each interval
        a = cut[:nsamples]
        b = cut[1:nsamples + 1]
        _center = (a + b)/2
        for j in range(dim):
            H[:, j] = np.random.permutation(_center)
    else:
        u = np.random.rand(nsamples, dim)
        for j in range(dim):
            H[:, j] = np.random.permutation(np.arange(nsamples))/nsamples + u[:,j]/nsamples
    
    return H


def sensitivity_analysis_lhs(nsteps=100, mi=10, Mo_lim=(0,100),
                             qi_lim=(0,1000), T0_lim=(5,60),
                             H_lim=(1., 10), ws_lim=(0,20), dt=1.):
    """
    Model sensitivity analysis using latin hypercube samples.
    """
    s = SynModel()
    samples = lhs(6, nsteps, centered=True)
    V = 8800
    Mo = samples[:,0] * (Mo_lim[1] - Mo_lim[0]) + Mo_lim[0]
    qi = samples[:,1] * (qi_lim[1] - qi_lim[0]) + qi_lim[0]
    T0 = samples[:,2] * (T0_lim[1] - T0_lim[0]) + T0_lim[0]
    X = 2.
    H = samples[:,3] * (H_lim[1] - H_lim[0]) + H_lim[0]
    ws = samples[:,4] * (ws_lim[1] - ws_lim[0]) + ws_lim[0]
    Mi = mi
    ydays = (samples[:,5] * 365).astype(int)
    dates = [date(2020, 1, 1) + timedelta(days=int(d)) for d in ydays]        
    results = np.zeros((nsteps, 12))
    columns = ['Qi', 'Mi', 'Mo', 'T0',
               'X', 'H', 'Ws', 'V', 'Date',
               'dT', 'dM', 'dX']
    for i in range(nsteps): 
        M = s.mass(V, T0[i])
        y = [T0[i], M, X, qi[i]*0.0864,
             Mi, Mo[i], H[i], ws[i]]
        dp = [0.] * 5
        y_new = s.fm.derivs(y, dates[i], dp, 1)
        results[i, 0] = qi[i]
        results[i, 1] = Mi
        results[i, 2] = Mo[i]
        results[i, 3] = T0[i]
        results[i, 4] = X
        results[i, 5] = H[i]
        results[i, 6] = ws[i]
        results[i, 7] = V
        results[i, 8] = ydays[i]
        results[i, 9] = y_new[0]
        results[i, 10] = y_new[1]
        results[i, 11] = y_new[2]

    return pd.DataFrame(results, columns=columns)


@cachier(stale_after=timedelta(weeks=2),
         cache_dir='.cache')
def sensitivity_analysis_grid(nsteps=10, mi=10, Mo_lim=(0, 90),
                             qi_lim=(0, 900), T0_lim=(5,50),
                             H_lim=(1, 10), ws_lim=(0,9), dt=1.):
    """
    Model sensitivity analysis using a regular grid.
    """
    s = SynModel()
    V = 8800
    Mo = np.linspace(Mo_lim[0], Mo_lim[1], nsteps)
    qi = np.linspace(qi_lim[0], qi_lim[1], nsteps)
    T0 = np.linspace(T0_lim[0], T0_lim[1], nsteps)
    X = 2.
    H = np.linspace(H_lim[0], H_lim[1], nsteps)
    ws = np.linspace(ws_lim[0], ws_lim[1], nsteps)
    Mi = mi
    ydays = np.linspace(0, 365, nsteps).astype(int)
    dates = [date(2020, 1, 1) + timedelta(days=int(d)) for d in ydays]        
    results = np.zeros((nsteps**6, 14))
    columns = ['Qi', 'Mi', 'Mo', 'T0',
               'X', 'H', 'Ws', 'V', 'Date',
               'dT', 'dM', 'dX', 'Me', 'Qe']
    i = 0
    for h in tqdm(H):
        for dt in dates:
            for w in ws:
                for T in T0:
                    for q in qi:
                        for mo in Mo:
                            M = s.mass(V, T)
                            y = [T, M, X, q*0.0864,
                                 Mi, mo, h, w]
                            dp = [0.] * 5
                            y_new = s.fm.derivs(y, dt, dp, 1)
                            qe, me = s.fm.evap
                            results[i, 0] = q
                            results[i, 1] = Mi
                            results[i, 2] = mo
                            results[i, 3] = T
                            results[i, 4] = X
                            results[i, 5] = h
                            results[i, 6] = w
                            results[i, 7] = V
                            results[i, 8] = dt.toordinal()
                            results[i, 9] = y_new[0]
                            results[i, 10] = y_new[1]
                            results[i, 11] = y_new[2]
                            results[i, 12] = me
                            results[i, 13] = qe
                            i += 1

    return pd.DataFrame(results, columns=columns)

