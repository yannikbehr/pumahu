"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""

from collections import defaultdict
import datetime
import os

import numpy as np
from numpy.ma import masked_equal
import pandas as pd
import pkg_resources
import xarray as xr

from sampling import (NestedSampling, Uniform,
                      Callback, Normal, Constant,
                      SamplingException)

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_continuous_white_noise

import progressbar

from clemb.forward_model import forward_model, fullness, esol, es


class NSCallback(Callback):
    """
    Callback function to compute the log-likelihood for nested sampling.
    """

    def __init__(self):
        Callback.__init__(self)

    def set_data(self, y1, cov, month, dt, ws):
        self.y1 = y1
        self.ws = ws
        # the precision matrix is the inverse of the
        # covariance matrix
        self.prec = np.linalg.inv(cov)
        self.factor = -np.log(np.power(2.*np.pi, cov.shape[0])
                              + np.sqrt(np.linalg.det(cov)))
        self.month = month
        self.dt = dt
        self.y_new = None

    def run(self, vals):
        try:
            Q_in = vals[0]*0.0864
            M_melt = vals[1]
            Mout = vals[2]
            H = vals[3]
            T = vals[4]
            M = vals[5]
            X = vals[6]
            a = vals[7]
            v = vals[8]
            y0 = np.array([T, M, X])
            solar = esol(self.dt, a, self.month)
            y_new, steam, mevap = forward_model(y0, self.dt, a, v, Q_in,
                                                M_melt, Mout, solar,
                                                H, self.ws, method='euler')
            self.y_new = y_new
            lh = self.factor - 0.5*np.dot(y_new-self.y1,
                                          np.dot(self.prec, y_new-self.y1))
        except:
            raise SamplingException("Oh no, a SamplingException!")
        return lh


def df_resample(df):
    """
    Resample dataframe to daily sample rate.
    """
    # First upsample to 15 min intervals combined with a linear interpolation
    ndates = pd.date_range(start=df.index.date[0], end=df.index.date[-1],
                           freq='15T')
    ndf = df.reindex(ndates, method='nearest',
                     tolerance=np.timedelta64(15, 'm')).interpolate()
    # Then downsample to 1 day intervals assigning the new values to mid day
    ndf = ndf.resample('1D', label='left').mean()
    return ndf


def common_date(date):
    """
    Function used for pandas group-by.
    If there are several measurements in
    one day, take the mean.
    """
    ndt = pd.Timestamp(year=date.year,
                       month=date.month,
                       day=date.day)
    return ndt


def interpolate_mg(df, dt=1):
    """
    Inter- and extrapolate Mg++ measurements using a
    non-linear Kalman filter.
    """

    def f_x(x, dt):
        """
        Forward model exponential decay
        """
        _k = x[1]/1e3
        _dt = dt
        _y = x[0]
        if isinstance(dt, np.ndarray):
            _dt = dt[0]
        # 4th order Runge-Kutta
        k0 = -_k * _y * _dt
        k1 = -_k * (_y + 0.5 * k0) * _dt
        k2 = -_k * (_y + 0.5 * k1) * _dt
        k3 = -_k * (_y + k2) * _dt
        _y_next = _y + 1./6.*(k0 + 2 * k1 + 2 * k2 + k3)
        return np.array([_y_next, x[1]])

    def h_x(x):
        """
        Measurement function
        """
        return [x[0]]

    dts = np.r_[0, np.cumsum(np.diff(df.index).astype(int)/(86400*1e9))]
    dts = dts[:, np.newaxis]
    ny = df['obs'].values
    ny = np.where(np.isnan(ny), None, ny)

    points = MerweScaledSigmaPoints(n=2, alpha=.01, beta=2., kappa=1.)
    kf = UKF(dim_x=2, dim_z=1, dt=dt, fx=f_x, hx=h_x, points=points)
    kf.x = np.array([ny[0], .6])
    kf.Q = Q_continuous_white_noise(2, dt=dt, spectral_density=1e-7)
    kf.P = np.diag([100.**2, 3.**2])
    kf.R = 50.**2
    npoints = dts.size
    means = np.zeros((npoints-1, 2))
    covariances = np.zeros((npoints-1, 2, 2))
    for i, z_n in enumerate(ny[1:]):
        kf.predict()
        kf.update(z_n)
        means[i, :] = kf.x
        covariances[i, :, :] = kf.P
    Ms, P, K = kf.rts_smoother(means, covariances)
    y_new = np.r_[ny[0], Ms[:, 0]]
    y_std = np.r_[50, np.sqrt(P[:, 0, 0])]
    return pd.DataFrame({'obs': y_new,
                         'obs_err': y_std,
                         'orig': df['obs'].values},
                        index=df.index)


def get_mg_data(tstart=None,
                tend=datetime.datetime.utcnow().strftime("%Y-%m-%d")):
    # Get Mg++ concentration
    url = "https://fits.geonet.org.nz/observation?siteID=RU003&typeID=Mg-w"
    names = ['obs', 'obs_err']
    mg_df = pd.read_csv(url, index_col=0, names=names, skiprows=1,
                        parse_dates=True)
    if tstart is not None:
        tstart = max(mg_df.index.min(), pd.Timestamp(tstart))
    else:
        tstart = mg_df.index.min()
    mg_df = mg_df.loc[(mg_df.index >= tstart) & (mg_df.index <= tend)]

    mg_df = mg_df.groupby(common_date, axis=0).mean()
    new_dates = pd.date_range(start=tstart, end=tend, freq='D')
    mg_df = mg_df.reindex(index=new_dates)
    # Find the first non-NaN entry
    tstart_min = mg_df.loc[~mg_df['obs'].isnull()].index[0]
    # Ensure the time series starts with a non-NaN value
    mg_df = mg_df.loc[mg_df.index >= tstart_min]
    return interpolate_mg(mg_df)


def interpolate_T(df, dt=1):
    dts = np.r_[0, np.cumsum(np.diff(df.index).astype(int)/(86400*1e9))]
    dts = dts[:, np.newaxis]
    ny = df['t'].values
    ny = np.where(np.isnan(ny), None, ny)
    kf = KF(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1], [0, 1]])
    kf.H = np.array([[1., 0]])
    if ny[1] is not None:
        dT0 = ny[1] - ny[0]
    else:
        dT0 = 0.
    kf.x = np.array([ny[0], dT0])
    kf.Q = Q_continuous_white_noise(2, dt=dt, spectral_density=3e-2)
    kf.P *= 1e-5**2
    kf.R = .5**2
    means, covariances, _, _ = kf.batch_filter(ny[1:])
    Ms, P, _, _ = kf.rts_smoother(means, covariances)
    y_new = np.r_[ny[0], Ms[:, 0]]
    y_std = np.r_[.3, np.sqrt(P[:, 0, 0])]
    return pd.DataFrame({'t': y_new,
                         't_err': y_std,
                         't_orig': df['t'].values},
                        index=df.index)


def get_T(tstart=None, tend=datetime.datetime.utcnow().strftime("%Y-%m-%d")):
    # Get temperature
    # Temperature has been recorded by 3 different sensors so 3 individual
    # requests have to be made
    url = "https://fits.geonet.org.nz/observation?"
    url += "siteID=RU001&typeID=t&methodID={}"
    names = ['t', 't_err']
    tdf1 = pd.read_csv(url.format('therm'),
                       index_col=0, names=names, skiprows=1,
                       parse_dates=True)
    tdf2 = pd.read_csv(url.format('thermcoup'),
                       index_col=0, names=names, skiprows=1,
                       parse_dates=True)
    tdf3 = pd.read_csv(url.format('logic'),
                       index_col=0, names=names, skiprows=1,
                       parse_dates=True)
    tdf3 = tdf3.combine_first(tdf2)
    t_df = tdf3.combine_first(tdf1)
    if tstart is not None:
        tstart = max(t_df.index.min(), pd.Timestamp(tstart))
    else:
        tstart = t_df.index.min()
    t_df = t_df.groupby(common_date, axis=0).mean()
    t_df = t_df.loc[(t_df.index >= tstart) & (t_df.index <= tend)]
    new_dates = pd.date_range(start=tstart, end=tend, freq='D')
    t_df = t_df.reindex(index=new_dates)
    # Find the first non-NaN entry
    tstart_min = t_df.loc[~t_df['t'].isnull()].index[0]
    # Ensure the time series starts with a non-NaN value
    t_df = t_df.loc[t_df.index >= tstart_min]
    return interpolate_T(t_df)


def interpolate_ll(df, dt=1):
    dts = np.r_[0, np.cumsum(np.diff(df.index).astype(int)/(86400*1e9))]
    dts = dts[:, np.newaxis]
    ny = df['h'].values
    ny = np.where(np.isnan(ny), None, ny)
    kf = KF(dim_x=1, dim_z=1)
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    if ny[1] is not None:
        dT0 = ny[1] - ny[0]
    else:
        dT0 = 0.
    kf.x = np.array([ny[0]])
    kf.Q = 1e-2**2
    kf.P = 0.03**2
    kf.R = 0.02**2
    means, covariances, _, _ = kf.batch_filter(ny[1:])
    Ms, P, _, _ = kf.rts_smoother(means, covariances)
    y_new = np.r_[ny[0], Ms[:, 0]]
    y_std = np.r_[0.03, np.sqrt(P[:, 0, 0])]
    return pd.DataFrame({'h': y_new,
                         'h_err': y_std,
                         'h_orig': df['h'].values},
                        index=df.index)


def get_ll(tstart=None, tend=datetime.datetime.utcnow().strftime("%Y-%m-%d")):
    # Get lake level
    # The lake level data is stored with respect to the overflow level of
    # the lake. Unfortunately, that level has changed over time so to get
    # the absolute lake level altitude, data from different periods have to
    # be corrected differently. Also, lake level data has been measured by
    # different methods requiring several requests.
    url = "https://fits.geonet.org.nz/observation?siteID={}&typeID=z"
    names = ['h', 'h_err']
    ldf = pd.read_csv(url.format('RU001'),
                      index_col=0, names=names, skiprows=1,
                      parse_dates=True)
    ldf1 = pd.read_csv(url.format('RU001A'),
                       index_col=0, names=names, skiprows=1,
                       parse_dates=True)
    ll_df = ldf.combine_first(ldf1)
    ll_df.loc[ll_df.index < '1997-01-01', 'h'] = 2530. + \
        ll_df.loc[ll_df.index < '1997-01-01', 'h']
    ll_df.loc[(ll_df.index > '1997-01-01') & (ll_df.index < '2012-12-31'), 'h'] = 2529.5 + \
              (ll_df.loc[(ll_df.index > '1997-01-01') & (ll_df.index < '2012-12-31'), 'h'] - 1.3)
    ll_df.loc[ll_df.index > '2016-01-01', 'h'] = 2529.35 + (ll_df.loc[ll_df.index > '2016-01-01', 'h'] - 2.0)
    
    if tstart is not None:
        tstart = max(ll_df.index.min(), pd.Timestamp(tstart))
    else:
        tstart = ll_df.index.min()
    ll_df = ll_df.groupby(common_date, axis=0).mean()
    ll_df = ll_df.loc[(ll_df.index >= tstart) & (ll_df.index <= tend)]
    new_dates = pd.date_range(start=tstart, end=tend, freq='D')
    ll_df = ll_df.reindex(index=new_dates)
    # Find the first non-NaN entry
    tstart_min = ll_df.loc[~ll_df['h'].isnull()].index[0]
    # Ensure the time series starts with a non-NaN value
    ll_df = ll_df.loc[ll_df.index >= tstart_min]
    return interpolate_ll(ll_df)


class LakeDataCSV:
    """
    Load the lake measurements from a CSV file.
    """

    def __init__(self, buf=None):
        if buf is not None:
            self._buf = buf
        else:
            self._buf = pkg_resources.resource_stream(
                __name__, 'data/data.dat')
        self.df = None

    def get_data(self, start, end):
        if self.df is None:
            rd = defaultdict(list)
            with self._buf:
                t0 = np.datetime64('2000-01-01')
                while True:
                    l = self._buf.readline()
                    if not l:
                        break
                    # ignore commented lines
                    l = l.decode()
                    if not l.startswith(' '):
                        continue
                    a = l.split()
                    y, m, d = map(int, a[0:3])
                    te, hgt, fl, img, icl, dr, oheavy, deut = map(float, a[3:])
                    dt = np.datetime64('{}-{:02d}-{:02d}'.format(y, m, d))
                    no = (dt - t0).astype(int) - 1
                    rd['date'].append(dt)
                    rd['nd'].append(no)
                    rd['t'].append(te)
                    rd['h'].append(hgt)
                    rd['f'].append(fl)
                    rd['o18'].append(oheavy)
                    rd['h2'].append(deut)
                    rd['o18m'].append(oheavy)
                    rd['h2m'].append(deut)
                    rd['m'].append(img / 1000.)
                    rd['c'].append(icl / 1000.)
                    rd['dv'].append(1.0)
            self.df = pd.DataFrame(rd, index=rd['date'])
            self.df = self.df.reindex(
                pd.date_range(start=self.df.index.min(),
                              end=self.df.index.max())).interpolate()
        start = max(self.df.index.min(), pd.Timestamp(start))
        end = min(self.df.index.max(), pd.Timestamp(end))
        df = self.df.loc[start:end]
        return df


def derived_obs(df1, df2, df3, nsamples=100):
    """
    Compute absolute amount of Mg++, volume, lake aread,
    water density and lake mass using Monte Carlo sampling
    """
    rn1 = np.random.randn(df1['obs'].size, nsamples)
    rn1 = rn1*np.tile(df1['obs_err'].values, (nsamples,1)).T + np.tile(df1['obs'].values, (nsamples,1)).T

    rn2 = np.random.randn(df3['h'].size, nsamples)
    rn2 = rn2*np.tile(df3['h_err'].values, (nsamples, 1)).T + np.tile(df3['h'].values, (nsamples,1)).T
    a, vol = fullness(rn2)
    X = rn1*vol*1.e-6
    
    p_mean = 1.003 - 0.00033 * df2['t'].values
    p_std = 0.00033*df2['t_err'].values
    
    rn3 = np.random.randn(p_mean.size, nsamples)
    rn3 = rn3*np.tile(p_std, (nsamples, 1)).T + np.tile(p_mean, (nsamples,1)).T
    M = rn3*vol
    return (X.mean(axis=1), X.std(axis=1), vol.mean(axis=1), vol.std(axis=1),
            p_mean, p_std, M.mean(axis=1), M.std(axis=1),
            a.mean(axis=1), a.std(axis=1))


class LakeDataFITS:
    """
    Load the lake measurements from the FITS database.
    """

    def __init__(self, url="https://fits.geonet.org.nz/observation"):
        self.base_url = url
        self.df = None

    def get_data(self, start, end, outdir='/tmp'):
        """
        Request data from the FITS database unless it has been already cached.
        """
        fn_out = os.path.join(outdir,
                              'measurements_{:s}_{:s}.h5'.format(start, end))
        if os.path.isfile(fn_out):
            self.df = pd.read_hdf(fn_out, 'table')

        if self.df is None:
            df1 = get_mg_data(tstart=start, tend=end)
            # Get temperature
            df2 = get_T(tstart=df1.index[0], tend=df1.index[-1])
            df3 = get_ll(tstart=df1.index[0], tend=df1.index[-1])
            # Find the timespan for which all three datasets have data
            tstart_max = max(max(df1.index[0], df2.index[0]), df3.index[0])
            tend_min = min(min(df1.index[-1], df2.index[-1]), df3.index[-1])
            (X, X_err, v, v_err, p, p_err,
             M, M_err, a, a_err) = derived_obs(df1, df2, df3, nsamples=20000)

            self.df = pd.DataFrame({'T': df2['t'],
                                    'T_err': df2['t_err'],
                                    'z': df3['h'],
                                    'z_err': df3['h_err'],
                                    'Mg': df1['obs'],
                                    'Mg_err': df1['obs_err'],
                                    'X': X,
                                    'X_err': X_err,
                                    'v': v,
                                    'v_err': v_err,
                                    'a': a,
                                    'a_err': a_err,
                                    'p': p,
                                    'p_err': p_err,
                                    'M': M,
                                    'M_err': M_err})
            self.df = self.df.loc[tstart_max:tend_min]
            self.df['dv'] = np.ones(self.df.index.size)
            self.df.to_hdf(fn_out, 'table')
        return self.df

    def to_table(self, start, end, filename):
        """
        Write a data file that can be read by the original Fortran code and
        used for unit testing.
        """
        d = self.get_data(start, end)
        lines = []
        fl = 0
        dr = 0.0
        o18 = 0.0
        deut = 0.0
        cl = 1
        for i in range(d['T'].size):
            date = d['T'].index[i]
            s_input = (date.year, date.month, date.day, d['T'].ix[i],
                       d['Z'].ix[i], fl, int(round(d['Mg'].ix[i])),
                       cl, dr, o18, deut)
            s_format = "{:<6d}{:<4d}{:<4d}{:<7.1f}{:<8.1f}{:<4d}{:<6d}{:<6d}"
            s_format += "{:<5.1f}{:<7.1f}{:<7.1f}\n"
            line = s_format.format(*s_input)
            lines.append(line)
        with open(filename, 'w') as fh:
            fh.writelines(lines)


class WindDataCSV:
    """
    Load wind speed data from a CSV file.
    """

    def __init__(self, buf=None, default=4.5):
        if buf is not None:
            self._buf = buf
        else:
            self._buf = pkg_resources.resource_stream(
                __name__, 'data/wind.dat')
        self._default = default
        self.sr = None

    def get_data(self, start, end):
        if self.sr is None:
            with self._buf:
                windspeed = []
                dates = []
                while True:
                    l = self._buf.readline()
                    if not l:
                        break
                    l = l.decode()
                    a = l.split()
                    y, m, d = map(int, a[0:3])
                    ws, wd = map(float, a[3:])
                    dates.append(
                        np.datetime64('{}-{:02d}-{:02d}'.format(y, m, d)))
                    windspeed.append(ws)
            self.sr = pd.Series(windspeed, index=dates)
        sr = self.sr.reindex(
            pd.date_range(start=start, end=end)).fillna(self._default)
        return sr


class Clemb:
    """
    Compute crater lake energy and mass balance. The model currently accounts
    for evaporation effects and melt water flow inferred from the dilution of
    magnesium and chloride ions.
    """

    def __init__(self, lakedata, winddata, start, end):
        """
        Load the lake data (temperature, lake level, concentration of Mg++,
        Cl-, O18 and deuterium) and the wind data.
        """
        self.lakedata = lakedata
        self.winddata = winddata
        self._df = lakedata.get_data(start, end)
        self._dates = self._df.index
        self._df['W'] = winddata.get_data(self._dates[0], self._dates[-1])
        self._df['H'] = np.ones(self._dates.size) * 6.0
        self.use_drmg = False
        # Specific heat for water
        self.cw = 0.0042
        self.results_dir = './'

    def get_variable(self, key):
        if key in self._ld:
            return self._ld[key]
        elif key.lower() == 'wind':
            return self._wd
        elif key.lower() == 'enthalpy':
            return self._enthalpy
        else:
            raise AttributeError('Unknown variable name.')

    def update_data(self, start, end):
        """
        Update the timeframe to analyse.
        """
        # allow for tinkering with the dilution factor
        old_dv = self._df['dv'].data

        self._ld = self.lakedata.get_data(start, end)
        self._wd = self.winddata.get_data(start, end)
        self._dates = self._ld.index

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
            self._df['dv'] = new_dv

    @property
    def drmg(self):
        return self.use_drmg

    @drmg.setter
    def drmg(self, val):
        self.use_drmg = val

    def run_backward(self, new=False):
        """
        Compute the amount of steam and energy that has to be put into a crater
        lake to cause an observed temperature change.
        """
        tstart = self._dates[0]
        tend = self._dates[-1]
        res_fn = os.path.join(self.results_dir,
                              'backward_{:s}_{:s}.nc')
        res_fn = res_fn.format(tstart.strftime('%Y-%m-%d'),
                               tend.strftime('%Y-%m-%d'))
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
        nd = (self._df.index[1:] - self._df.index[:-1]).days
        # time interval in Megaseconds
        timem = 0.0864 * nd
        density = 1.003 - 0.00033 * self._df['T']
        a, vol = fullness(self._df['z'].values)

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
        loss, ev = es(self._df['T'][:-1].values,
                      self._df['W'][:-1].values, a[:-1])

        # Energy balances [TJ];
        # es = Surface heat loss, el = change in stored energy
        e = loss + self.el(self._df['T'][:-1].values,
                           self._df['T'][1:].values, vol[:-1])

        # e is energy required from steam, so is reduced by sun energy
        e -= esol((self._df.index[1:] - self._df.index[:-1]).days, a[:-1],
                  self._dates[:-1].month)
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
        res.to_netcdf(res_fn)
        return res

    def run_forward(self, nsamples=10000, nresample=500, new=False):
        tstart = self._dates[0]
        tend = self._dates[-1]
        res_fn = os.path.join(self.results_dir,
                              'forward_{:s}_{:s}.nc')
        res_fn = res_fn.format(tstart.strftime('%Y-%m-%d'),
                               tend.strftime('%Y-%m-%d'))
        if not new and os.path.isfile(res_fn):
            res = xr.open_dataset(res_fn)
            res.close()
            return res

        nsteps = self._df.shape[0] - 1
        dt = 1.
        nparams = 9

        qin = Uniform('qin', 0, 1000)
        m_in = Uniform('m_in', 0, 20)
        h = Constant('h', 6.)
        m_out = Uniform('m_out', 0, 20)
        ws = 4.5

        # return values
        qin_samples = np.zeros((nsteps, nresample))
        m_in_samples = np.zeros((nsteps, nresample))
        m_out_samples = np.zeros((nsteps, nresample))
        h_samples = np.zeros((nsteps, nresample))
        lh = np.zeros((nsteps, nresample))
        exp = np.zeros((nsteps, nparams))
        var = np.zeros((nsteps, nparams))
        model_data = np.zeros((nsteps, nresample, 3))
        steam = np.zeros((nsteps, nresample))
        mevap = np.zeros((nsteps, nresample))

        ns = NestedSampling()
        pycb = NSCallback()
        pycb.__disown__()
        ns.setCallback(pycb)

        with progressbar.ProgressBar(max_value=nsteps-1) as bar:
            for i in range(nsteps):
                # Take samples from the input
                T = Normal('T', self._df['T'][i], self._df['T_err'][i])
                M = Normal('M', self._df['M'][i], self._df['M_err'][i])
                X = Normal('X', self._df['X'][i], self._df['X_err'][i])
                v = Normal('v', self._df['v'][i], self._df['v_err'][i])
                a = Normal('a', self._df['a'][i], self._df['a_err'][i])
                T_sigma = self._df['T_err'][i+1]
                M_sigma = self._df['M_err'][i+1]
                X_sigma = self._df['X_err'][i+1]
                cov = np.array([[T_sigma*T_sigma, 0., 0.],
                                [0., M_sigma*M_sigma, 0.],
                                [0., 0., X_sigma*X_sigma]])
                T_next = self._df['T'][i+1]
                M_next = self._df['M'][i+1]
                X_next = self._df['X'][i+1]

                y = np.array([T, M, X])
                y_next = np.array([T_next, M_next, X_next])
                pycb.set_data(y_next, cov, self._dates[i].month, dt, ws)
                rs = ns.explore(vars=[qin, m_in, m_out, h, T, M, X, a, v],
                                initial_samples=100,
                                maximum_steps=nsamples)
                del T, M, X, v, a
                smp = rs.resample_posterior(nresample)
                exp[i, :] = rs.getexpt()
                var[i, :] = rs.getvar()
                for j, _s in enumerate(smp):
                    Q_in = _s._vars[0].get_value()
                    M_in = _s._vars[1].get_value()
                    M_out = _s._vars[2].get_value()
                    H = _s._vars[3].get_value()
                    T = _s._vars[4].get_value()
                    M = _s._vars[5].get_value()
                    X = _s._vars[6].get_value()
                    a = _s._vars[7].get_value()
                    v = _s._vars[8].get_value()
                    y = np.array([T, M, X])
                    solar = esol(dt, a, self._dates[i].month)
                    y_mod, st, me = forward_model(y, dt, a, v, Q_in*0.0864,
                                                  M_in, M_out, solar, H, ws)
                    steam[i, j] = st
                    mevap[i, j] = me
                    model_data[i, j, :] = y_mod
                    qin_samples[i, j] = Q_in
                    m_in_samples[i, j] = M_in
                    m_out_samples[i, j] = M_out
                    h_samples[i, j] = H
                    lh[i, j] = np.exp(_s._logL)
                del smp
                bar.update(i)
        res = xr.Dataset({'exp': (('dates', 'parameters'), exp),
                          'var': (('dates', 'parameters'), var),
                          'q_in': (('dates', 'sampleidx'),
                                   masked_equal(qin_samples, 0)),
                          'h': (('dates', 'sampleidx'),
                                masked_equal(h_samples, 0)),
                          'lh': (('dates', 'sampleidx'),
                                 masked_equal(lh, 0)),
                          'm_in': (('dates', 'sampleidx'),
                                   masked_equal(m_in_samples, 0)),
                          'm_out': (('dates', 'sampleidx'),
                                    masked_equal(m_out_samples, 0)),
                          'steam': (('dates', 'sampleidx'),
                                    masked_equal(steam, 0)),
                          'mevap': (('dates', 'sampleidx'),
                                    masked_equal(mevap, 0)),
                          'model': (('dates', 'sampleidx', 'obs'),
                                    masked_equal(model_data, 0))},
                         {'dates': self._dates[:-1],
                          'parameters': ['q_in', 'm_in', 'm_out',
                                         'h', 'T', 'M', 'X', 'a', 'v'],
                          'obs': ['T', 'M', 'X']})
        res.to_netcdf(res_fn)
        return res

    def el(self, t1, t2, vol):
        """
        Change in Energy stored in the lake [TJ]
        """
        return (t2 - t1) * vol * self.cw
