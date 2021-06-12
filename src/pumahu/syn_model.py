import hashlib
from datetime import date, timedelta

from cachier import cachier
import numpy as np
import pandas as pd
from scipy.signal import tukey
from scipy.stats import gamma
import xarray as xr
from tqdm import tqdm

from .forward_model import Forwardmodel


def setup_test(q_in=1000.):
    """
    A synthetic scenario for unit testing.

    :param q_in: Maximum heat input rate [MW]
    :type q_in: float
    :returns: An xarray DataFrame containing the input
              data for the synthetic model
    :rtype: :class:`xarray.DataArray`
    """
    dates = pd.date_range(start='2017-01-01', end='2017-01-04',
                          freq='D')
    data = np.zeros((4,4))
    params = ['q_in', 'm_in', 'h', 'W']
    # qi
    data[0, :] = np.array([0., .2, .6, 1.])*q_in
    # Mi
    data[1, :] = np.ones(4)*10.
    # H
    data[2, :] = np.ones(4)*6
    # W
    data[3, :] = np.ones(4)*4.5
    return xr.DataArray(data, coords=[('params', params),
                                      ('times', dates)])


def setup_realistic(q_in=1000., tmax=90., sinterval=10):
    """
    Synthetic scenario where the heat influx is a
    combination of 3rd order polynomial and sinusoid.

    :param q_in: Maximum heat input rate [MW]
    :type q_in: float
    :param tmax: Length of the time-series [days]
    :type tmax: float
    :param sinterval: Sampling interval [mins]
    :type sinterval: int
    :returns: An xarray DataFrame containing the input
              data for the synthetic model
    :rtype: :class:`xarray.DataArray`
    """
    dt = (sinterval * 60)/86400.
    t = np.arange(0,tmax+dt, dt)
    dates = pd.date_range(start='2020-03-03', freq='%dmin'%sinterval,
                          periods=t.size)
    a = 1.
    b = -20
    c = b**2/4+3
    y = 1./(a*t**2 + b*t + c)
    y = q_in*y/max(y)

    npts = y.size
    y_last = y[npts//2]
    f = 1./20.
    t1 = t[npts//2::]
    y[npts//2:] = np.sin(2.*np.pi*f*t1 + 2*np.pi)*q_in/4.+q_in/4.
    y[npts//2:] *= tukey(t1.size, 0.9)
    y[npts//2:] += y_last
    
    data = np.zeros((4,t.size))
    params = ['q_in', 'm_in', 'h', 'W']
    # qi
    data[0, :] = y
    # Mi
    data[1, :] = np.ones(t.size)*10.
    # H
    data[2, :] = np.ones(t.size)*6
    # W
    data[3, :] = np.ones(t.size)*4.5
    return xr.DataArray(data, coords=[('params', params),
                                      ('times', dates)])
#        if mode == 'sinus':
#            qi = np.sin(2.*np.pi*self.f*t + np.deg2rad(15))*q_in/2.+q_in/2.
#            qi *= tukey(nsteps, 0.9)
#        if mode == 'tukey':
#            tail = int(10/dt)
#            qi = np.ones(nsteps-tail)*q_in
#            qi *= tukey(nsteps-tail, 0.9)
#            qi_tmp = np.zeros(nsteps)
#            qi_tmp[0:nsteps-tail] = qi
#            qi = qi_tmp
#        if mode == 'step':
#            tail = int(10/dt)
#            qi = np.ones(nsteps-tail)*q_in
#            qi_tmp = np.zeros(nsteps)
#            qi_tmp[0:nsteps-tail] = qi
#            qi = qi_tmp
#        if mode == 'constant':
#            qi = np.ones(nsteps)*q_in
#        if mode == 'gamma':
#            y = gamma.pdf(t, 3., loc=4)
#            y /= y.max()
#            y *= q_in
#            qi = y
#
#        if mode == 'observed':
#            # a polynomial model of a real inversion
#            z = np.array([-6.93795125e-30,  1.17673746e-26, -9.19940930e-24,
#                          4.39229128e-21, -1.43037875e-18,  3.36004438e-16,
#                          -5.87098984e-14,  7.75486512e-12, -7.79213224e-10,
#                          5.94657977e-08, -3.41657063e-06,  1.45392771e-04,
#                          -4.46901292e-03,  9.56540064e-02, -1.35189993e+00,
#                          1.16488040e+01, -5.37668791e+01,  1.05017651e+02,
#                          2.91772946e+01])
#            p = np.poly1d(z)
#            dates = pd.date_range(start='2018-4-1', end='2018-9-30',
#                                  periods=183)
#            t = np.r_[0, np.cumsum(np.diff(dates)/np.timedelta64(1, 'D'))]
#            qi = p(t)
#            nsteps = 183
#
#        if mode == 'test':
#            qi = np.array([0., .2, .6, 1.])*q_in
#            dates = pd.date_range(start='2017-01-01', end='2017-01-04',
#                                  freq='D')
#            nsteps = 4
#            
def myhash(args, kwds):
    key = []
    key.append(args[0].integration_method)
    for a in args[1:]:
        if isinstance(a, xr.core.dataarray.DataArray):
            key.append(hashlib.sha256(a.values).hexdigest())
        elif isinstance(a, list):
            key.append(tuple(a))
        else:
            key.append(a)

    key = tuple(key)
    key += tuple(sorted(kwds.items()))
    return key


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
        self.integration_method=integration_method
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

    @cachier(stale_after=timedelta(weeks=2),
             cache_dir='.cache', hash_params=myhash)
    def run(self, data, gradient=False, addnoise=False, 
            estimatenoise=False):
        """
        Produce synthetic observations.
        """
        prms = ['T', 'M', 'X', 'q_in',
                'm_in', 'm_out', 'h', 'W',
                'p', 'A', 'V', 'z', 'Mg']
        nprms = len(prms)
        nsteps = data.times.size
        dates = pd.to_datetime(data['times'].values)
        qi = data.loc[dict(params='q_in')].values
        exp = np.zeros((nsteps, nprms, 2))
        error = {'T': 0.4, 'z': 0.01, 'M': 2.,
                 'X': 0.4, 'A': 30., 'V': 2.0,
                 'Mg': 50, 'm_out': .25, 'W':.5,
                 'h': .01}
 
        V = 8800
        A = self.a
        X = 2.
        M = self.mass(V, self.T0)
        ll = self.level(V, A)
        Mo = self.outflow(ll)
        exp[0, 0:3, 0] = [self.T0, M, X]
        exp[0, prms.index('V'), 0] = V 
        exp[0, prms.index('z'), 0] = ll
        exp[0, prms.index('A'), 0] = A 
        Mg = exp[0, prms.index('X'), 0]/exp[0, prms.index('M'), 0]*1e6
        exp[0, prms.index('Mg'), 0] = Mg
        p = 1.003 - 0.00033 * exp[0, prms.index('T'), 0]
        exp[0, prms.index('p'), 0] = p
        
        for i in range(nsteps - 1):
            exp[i, prms.index('h'), 0] = data.isel(times=i).loc['h']
            exp[i, prms.index('W'), 0] = data.isel(times=i).loc['W'] 
            exp[i, prms.index('m_in'), 0] = data.isel(times=i).loc['m_in'] 
            exp[i, prms.index('q_in'), 0] = 0.0864*qi[i]
            exp[i, prms.index('m_out'), 0] = Mo
            dt = (dates[i+1] - dates[i])/pd.Timedelta('1D')

            if not gradient:
                dp = [0.] * 5
            else:
                dp = [0.] * 5
                dp[0] = (qi[i+1] - qi[i])*0.0864/dt
            y_new = self.fm.integrate(exp[i, 0:8, 0],
                                      dates[i], dt, dp)
            V = self.volume(y_new[1], y_new[0])
            ll = self.level(V, A)
            Mo = self.outflow(ll)
            exp[i+1, 0:8, 0] = y_new.copy()
            exp[i+1, prms.index('V'), 0] = V 
            exp[i+1, prms.index('z'), 0] = ll
            exp[i+1, prms.index('A'), 0] = A 
            Mg = exp[i+1, prms.index('X'), 0]/exp[i+1, prms.index('M'), 0]*1e6
            exp[i+1, prms.index('Mg'), 0] = Mg
            p = 1.003 - 0.00033 * exp[i+1, prms.index('T'), 0]
            exp[i+1, prms.index('p'), 0] = p
            
            # Prescripe errors
            for k in error.keys(): 
                exp[i+1, prms.index(k), 1] = error[k]
        exp[:, prms.index('q_in'), 0] /= 0.0864

        if addnoise:
            for k in ['T', 'M', 'X', 'W', 'h']:
                exp[:, prms.index(k), 0] += np.random.normal(scale=error[k],
                                                             size=nsteps)

        xds = xr.Dataset({'exp': (('dates', 'parameters', 'val_std'), exp)},
                         {'dates': dates.values,
                          'parameters': prms,
                          'val_std': ['val', 'std']})
        return xds


def resample(xds, parameters=['T', 'M', 'X', 'z', 'W', 'h'], sinterval='1D'):
    """
    Resample a synthetic model to a given sampling interval.
    
    Parameters:
    -----------
    xds : :class:`xarray.Dataset`
          The input synthetic dataset.
    parameters : array_like
                 Parameters to resample
    sinterval : str
                The resampling interval. Currently, only downsampling
                supported.
    """
    nxds = xds.loc[dict(parameters=parameters, val_std='val')].resample(dates=sinterval)
    mn = nxds.mean().to_array().values
    st = nxds.std().to_array().values
    new_data = np.concatenate((mn[:,:,:,np.newaxis], st[:,:,:,np.newaxis]), axis=3)
    rxds = xr.DataArray(new_data[0], dims=('dates', 'parameters', 'val_std'),
                        coords=(nxds.mean().dates, parameters,['val', 'std']) )
    # fill fields with zero variance with the average
    # variance for the respective parameter
    idx = np.where(rxds.loc[dict(val_std='std')] == 0.)
    rxds[idx[0], idx[1], 1] = rxds[:, :, 1].mean(axis=0)
    return rxds


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

