from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from functools import lru_cache
import inspect
import os
import pkg_resources

from cachier import cachier
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_continuous_white_noise
import xarray as xr

from . import Forwardmodel, get_data

def fits_hash(args, kwargs):
    key = []
    for _a in args[1:]:
        key.append(_a)
    key = tuple(key)    
    key += tuple(sorted(kwargs.items()))
    hashValue = hash(key)
    return hashValue

class LakeData:
    """
    Load the lake measurements from the FITS database.
    """

    def __init__(self, url="https://fits.geonet.org.nz/observation",
                 csvfile=None, windspeed=5., m_out=10.,
                 enthalpy=3.):
        """
        Parameters:
        -----------
        url : str
              FITS url to retrieve data from
        csvfile : str
                  If not None, read data from a csv file instead of
                  requesting it from FITS
        windspeed : float
                  Default windspeed [m/s].
        enthalpy : float
                   Default enthalpy [MJ/kg/day]
        m_out : Default outflow rate [kt/day]
        """
        self.base_url = url
        self.xdf = None
        self.ws = windspeed
        self.ws_err = 0.5
        self.m_out = m_out
        self.m_out_err = .25
        self.h = enthalpy
        self.h_err = 0.01
        self.prms = ['T', 'z', 'Mg', 'X', 'V',
                     'A', 'p', 'M', 'dV', 'h', 
                     'W', 'm_out']
        if csvfile is None:
            self.get_data = self.get_data_fits
        else:
            self.csvfile = csvfile
            self.get_data = self.get_data_csv


    @cachier(stale_after=timedelta(weeks=2),
             cache_dir='~/.cache', hash_params=fits_hash)
    def get_data_fits(self, start, end, smoothing='kf'):
        """
        Request data from the FITS database unless it has been already cached.
        
        Parameters:
        -----------
        start : str
                The start date as a ISO 8601 compliant string.
        end : str
              The end date as a ISO 8601 compliant string.
        smoothing : str
                    Can be either 'kf' for Kalman Filter smoothing
                    or 'dv' to smooth by averaging over all values
                    in a day.
        
        Returns:
        --------
        :class:`xarray.DataArray`
            The observational data.
        """
        df1 = self.get_Mg(tstart=start, tend=end,
                          smoothing=smoothing)
        # Get temperature
        df2 = self.get_T(tstart=df1.index[0], tend=df1.index[-1],
                         smoothing=smoothing)
        df3 = self.get_ll(tstart=df1.index[0], tend=df1.index[-1],
                          smoothing=smoothing)
        # Find the timespan for which all three datasets have data
        tstart_max = max(max(df1.index[0], df2.index[0]), df3.index[0])
        tend_min = min(min(df1.index[-1], df2.index[-1]), df3.index[-1])
        (X, X_err, v, v_err, p, p_err,
         M, M_err, a, a_err) = self.derived_obs(df1, df2, df3,
                                                nsamples=20000)
        dsize = X.size
        result = np.zeros((dsize, len(self.prms), 2))*np.nan
        result[:, self.prms.index('T'), 0] = df2['t'].values
        result[:, self.prms.index('T'), 1] = df2['t_err'].values
        result[:, self.prms.index('z'), 0] = df3['h'].values
        result[:, self.prms.index('z'), 1] = df3['h_err'].values
        result[:, self.prms.index('Mg'), 0] = df1['Mg'].values
        result[:, self.prms.index('Mg'), 1] = df1['Mg_err'].values
        result[:, self.prms.index('X'), 0] = X
        result[:, self.prms.index('X'), 1] = X_err
        result[:, self.prms.index('V'), 0] = v
        result[:, self.prms.index('V'), 1] = v_err
        result[:, self.prms.index('A'), 0] = a
        result[:, self.prms.index('A'), 1] = a_err
        result[:, self.prms.index('p'), 0] = p
        result[:, self.prms.index('p'), 1] = p_err
        result[:, self.prms.index('M'), 0] = M
        result[:, self.prms.index('M'), 1] = M_err
        result[:, self.prms.index('dV'), 0] = np.ones(dsize)
        result[:, self.prms.index('dV'), 1] = np.ones(dsize)
        result[:, self.prms.index('W'), 0] = np.ones(dsize)*self.ws
        result[:, self.prms.index('W'), 1] = np.ones(dsize)*self.ws_err
        result[:, self.prms.index('h'), 0] = np.ones(dsize)*self.h
        result[:, self.prms.index('h'), 1] = np.ones(dsize)*self.h_err
        result[:, self.prms.index('m_out'), 0] = np.ones(dsize)*self.m_out
        result[:, self.prms.index('m_out'), 1] = np.ones(dsize)*self.m_out_err


        self.xdf = xr.DataArray(result,
                                dims=('dates', 'parameters', 'val_std'),
                                coords=(df1.index, self.prms,
                                        ['val', 'std']))
        self.xdf = self.xdf.loc[tstart_max:tend_min]
        return self.xdf

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

    def relative2absolutell(self, ll_df):
        """
        Convert relative to absolute (m a.s.l) lake level.
        
        Parameter:
        ----------
        ll_df : :class:`pandas.DataFrame`
                Relative lake level in meters above the sensor.
        
        Returns:
        --------
        :class:`pandas.DataFrame`
        Absolute lake level in meters a.s.l.
        """
        t1 = '1997-01-01'
        t2 = '2012-12-31'
        t3 = '2016-01-01'
        ll_df.loc[ll_df.index < t1, 'obs'] = 2530. + \
            ll_df.loc[ll_df.index < t1, 'obs']
        ll_df.loc[(ll_df.index > t1) & (ll_df.index < t2), 'obs'] = 2529.5 + (ll_df.loc[(ll_df.index > t1) & (ll_df.index < t2), 'obs'] - 1.3)
        ll_df.loc[ll_df.index > t3, 'obs'] = 2529.35 + (ll_df.loc[ll_df.index > t3, 'obs'] - 2.0)
        return ll_df
    
    @lru_cache(maxsize=4)
    def FITS_request(self, obs, lake='RCL'):
        """
        Request measurements from FITS.

        Parameter
        ---------
        obs : string
              The observation type to request. Can be either 'T' for
              temperature, 'L' for water level, or 'Mg' for Mg++
              concentration.
        lake : string
               The lake for which to request the data. Currently the
               only option is 'RCL' for Ruapehu Crater Lake.

        Returns
        -------
        df : pandas.DataFrame
             Returns a dataframe with columns 'obs' and 'obs_err' and
             the date-time of the observation as the index.
        """
        if lake == 'RCL':
            if obs == 'T':
                # Temperature has been recorded by 3 different sensors so
                # 3 individual requests have to be made
                url = self.base_url+"?siteID=RU001&typeID=t&methodID={}"
                names = ['obs', 'obs_err']
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
                tdf3 = tdf3.combine_first(tdf1)
                tdf3 = tdf3.tz_localize(None)
                return tdf3

            if obs == 'L':
                # Get lake level The lake level data is stored with respect to
                # the overflow level of the lake. Unfortunately, that level has
                # changed over time so to get the absolute lake level altitude,
                # data from different periods have to be corrected differently.
                # Also, lake level data has been measured by different methods
                # requiring several requests.
                url = self.base_url+"?siteID={}&typeID=z"
                names = ['obs', 'obs_err']
                ldf = pd.read_csv(url.format('RU001'),
                                  index_col=0, names=names, skiprows=1,
                                  parse_dates=True)
                ldf1 = pd.read_csv(url.format('RU001A'),
                                   index_col=0, names=names, skiprows=1,
                                   parse_dates=True)
                ll_df = ldf.combine_first(ldf1)
                ll_df = ll_df.tz_localize(None)
                return self.relative2absolutell(ll_df)

            if obs == 'Mg':
                # Get Mg++ concentration
                url = self.base_url+"?siteID=RU003&typeID=Mg-w"
                names = ['obs', 'obs_err']
                mg_df = pd.read_csv(url, index_col=0, names=names, skiprows=1,
                                    parse_dates=True)
                mg_df = mg_df.tz_localize(None)
                return mg_df

    def interpolate_mg(self, df, dt=1):
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
        ny = df['Mg'].values
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
        return pd.DataFrame({'Mg': y_new,
                             'Mg_err': y_std,
                             'Mg_orig': df['Mg'].values},
                            index=df.index)

    def get_Mg(self, tstart=None, tend=datetime.utcnow(),
               smoothing='kf'):
        """
        Get Mg++ measurements and estimate the measurement error.

        Parameters
        ----------
        tstart : string
                 The start date of the time series following the format
                 'yyyy-mm-dd'. The effective start of the time series will be
                 either tstart or the first non-NAN entry after tstart.
        tend : string
               The end date of the time series following the same format
               as tstart.
        smoothing : string
                    Defines the smoothing method to estimate the mean and
                    uncertainty. This can be either 'kf' for Kalman filter or
                    'dv' for daily variation. The latter computes the daily
                    mean and standard deviation.
        """
        df = self.FITS_request('Mg')
        df.rename(columns={'obs': 'Mg', 'obs_err': 'Mg_err'}, inplace=True)
        if tstart is not None:
            # Find the sampling date that is closest to the requested
            # start time
            idx = np.abs((df.index - pd.Timestamp(tstart))).argmin()
            _tstart = df.index[idx]
        else:
            _tstart = df.index.min()

        # make sure the start time is at midnight
        _tstart = pd.Timestamp(year=_tstart.year,
                               month=_tstart.month,
                               day=_tstart.day)
        df = df.loc[(df.index >= _tstart) & (df.index <= tend)]
        new_dates = pd.date_range(start=_tstart, end=tend, freq='D')
        if smoothing == 'kf':
            mg_df = df.groupby(pd.Grouper(freq='D')).mean()
            mg_df = mg_df.reindex(index=new_dates)
            img_df = self.interpolate_mg(mg_df)
            img_df = img_df.loc[(img_df.index >= _tstart) & (img_df.index <= tend)]
            return img_df
        elif smoothing == 'dv':
            # First compute the expected variance from the days
            # when we had two samples taken
            dates = []
            stds = {}
            for i in range(1, df.shape[0]):
                d0 = df.iloc[i-1].name.date()
                d1 = df.iloc[i].name.date()
                if d1 == d0:
                    stds[d1] = np.std([df.iloc[i-1]['Mg'],
                                       df.iloc[i]['Mg']])
                    dates.append(d1)
            stds_mean = np.mean(list(stds.values()))
            mg_df = df.groupby(pd.Grouper(freq='D'), axis=0).mean()
            for d in mg_df.index:
                dt = d.date()
                if dt in dates:
                    mg_df.loc[str(dt), 'Mg_err'] = stds[dt]
                else:
                    mg_df.loc[str(dt), 'Mg_err'] = stds_mean
            mg_df = mg_df.reindex(index=new_dates)
            return mg_df
        else:
            raise ValueError('smoothing has to be either "kf" or "dv".')

    def interpolate_T(self, df, dt=1):
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

    def get_T(self, tstart=None, tend=datetime.utcnow(),
              smoothing='kf'):
        """
        Get temperature measurements from FITS and estimate measurement errors.

        Parameters
        ----------
        tstart : string
                 The start date of the time series following the format
                 'yyyy-mm-dd'. The effective start of the time series will be
                 either tstart or the first non-NAN entry after tstart.
        tend : string
               The end date of the time series following the same format
               as tstart.
        smoothing : string
                    Defines the smoothing method to estimate the mean and
                    uncertainty. This can be either 'kf' for Kalman filter or
                    'dv' for daily variation. The latter computes the daily
                    mean and standard deviation.
        """
        df = self.FITS_request('T')
        if tstart is not None:
            tstart = max(df.index.min(), pd.Timestamp(tstart))
        else:
            tstart = df.index.min()
        df.rename(columns={'obs': 't', 'obs_err': 't_err'}, inplace=True)
        g = df['t'].groupby(pd.Grouper(freq='D'))
        _mean = g.mean()
        _std = g.std()
        # get index of all days that have less than 2 values
        idx = g.count() < 2
        _mean[idx] = np.nan
        _std[idx] = np.nan

        t_df = pd.DataFrame({'t': _mean, 't_err': _std},
                            index=_mean.index)
        t_df = t_df.loc[(t_df.index >= tstart) & (t_df.index <= tend)]
        new_dates = pd.date_range(start=tstart, end=tend, freq='D')
        t_df = t_df.reindex(index=new_dates)
        # Find the first non-NaN entry
        tstart_min = t_df.loc[~t_df['t'].isnull()].index[0]
        # Ensure the time series starts with a non-NaN value
        t_df = t_df.loc[t_df.index >= tstart_min]
        if smoothing == 'kf':
            return self.interpolate_T(t_df)
        elif smoothing == 'dv':
            return t_df
        else:
            raise ValueError('smoothing has to be either "kf" or "dv".')

    def interpolate_ll(self, df, dt=1):
        dts = np.r_[0, np.cumsum(np.diff(df.index).astype(int)/(86400*1e9))]
        dts = dts[:, np.newaxis]
        ny = df['h'].values
        ny = np.where(np.isnan(ny), None, ny)
        kf = KF(dim_x=1, dim_z=1)
        kf.F = np.array([[1.]])
        kf.H = np.array([[1.]])
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

    def get_ll(self, tstart=None, tend=datetime.utcnow(),
               smoothing='kf'):
        """
        Get lake level measurements from FITS and estimate measurement errors.

        Parameters
        ----------
        tstart : string
                 The start date of the time series following the format
                 'yyyy-mm-dd'. The effective start of the time series will be
                 either tstart or the first non-NAN entry after tstart.
        tend : string
               The end date of the time series following the same format
               as tstart.
        smoothing : string
                    Defines the smoothing method to estimate the mean and
                    uncertainty. This can be either 'kf' for Kalman filter or
                    'dv' for daily variation. The latter computes the daily
                    mean and standard deviation.
        """
        df = self.FITS_request('L')
        df.rename(columns={'obs': 'h', 'obs_err': 'h_err'}, inplace=True)
        if tstart is not None:
            tstart = max(df.index.min(), pd.Timestamp(tstart))
        else:
            tstart = df.index.min()
        g = df['h'].groupby(pd.Grouper(freq='D'))
        _mean = g.mean()
        _std = g.std()
        # get index of all days that have less than 2 values
        idx = g.count() < 2
        _mean[idx] = np.nan
        _std[idx] = np.nan

        ll_df = pd.DataFrame({'h': _mean, 'h_err': _std},
                             index=_mean.index)
        ll_df = ll_df.loc[(ll_df.index >= tstart) & (ll_df.index <= tend)]
        new_dates = pd.date_range(start=tstart, end=tend, freq='D')
        ll_df = ll_df.reindex(index=new_dates)
        # Find the first non-NaN entry
        tstart_min = ll_df.loc[~ll_df['h'].isnull()].index[0]
        # Ensure the time series starts with a non-NaN value
        ll_df = ll_df.loc[ll_df.index >= tstart_min]
        if smoothing == 'kf':
            return self.interpolate_ll(ll_df)
        elif smoothing == 'dv':
            return ll_df
        else:
            raise ValueError('smoothing has to be either "kf" or "dv".')

    def get_data_csv(self, start, end, buf=None):
        """
        Load the lake measurements from a CSV file.
        """
        if buf is not None:
            _buf = buf
        else:
            _buf = pkg_resources.resource_stream(
                    __name__, 'data/data.dat')
        df = None

        if df is None:
            rd = defaultdict(list)
            with _buf:
                t0 = np.datetime64('2000-01-01')
                while True:
                    l = _buf.readline()
                    if not l:
                        break
                    # ignore commented lines
                    try:
                        l = l.decode()
                    except AttributeError:
                        pass
                    if not l.startswith(' '):
                        continue
                    a = l.split()
                    y, m, d = map(int, a[0:3])
                    te, hgt, fl, img, icl, dr, oheavy, deut = map(float, a[3:])
                    dt = np.datetime64('{}-{:02d}-{:02d}'.format(y, m, d))
                    no = (dt - t0).astype(int) - 1
                    rd['date'].append(dt)
                    rd['nd'].append(no)
                    rd['T'].append(te)
                    rd['z'].append(hgt)
                    rd['f'].append(fl)
                    rd['o18'].append(oheavy)
                    rd['h2'].append(deut)
                    rd['o18m'].append(oheavy)
                    rd['h2m'].append(deut)
                    rd['Mg'].append(img / 1000.)
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

    def derived_obs(self, df1, df2, df3, nsamples=100, seed=42):
        """
        Compute absolute amount of Mg++, volume, lake area,
        water density and lake mass using Monte Carlo sampling
        """
        rs = np.random.default_rng(seed)
        rn1 = rs.normal(size=df1['Mg'].size * nsamples)
        rn1 = rn1.reshape(df1['Mg'].size, nsamples)
        rn1 = rn1*np.tile(df1['Mg_err'].values,
                          (nsamples, 1)).T + np.tile(df1['Mg'].values,
                                                     (nsamples, 1)).T

        rn2 = rs.normal(size=df3['h'].size * nsamples)
        rn2 = rn2.reshape(df3['h'].size, nsamples)
        rn2 = rn2*np.tile(df3['h_err'].values,
                          (nsamples, 1)).T + np.tile(df3['h'].values,
                                                     (nsamples, 1)).T
        fm = Forwardmodel()
        a, vol = fm.fullness(rn2)
        X = rn1*vol*1.e-6

        p_mean = 1.003 - 0.00033 * df2['t'].values
        p_std = 0.00033*df2['t_err'].values

        rn3 = rs.normal(size=p_mean.size * nsamples)
        rn3 = rn3.reshape(p_mean.size, nsamples)
        rn3 = rn3*np.tile(p_std, (nsamples, 1)).T + np.tile(p_mean,
                                                            (nsamples, 1)).T
        M = rn3*vol
        return (X.mean(axis=1), X.std(axis=1), vol.mean(axis=1),
                vol.std(axis=1), p_mean, p_std, M.mean(axis=1),
                M.std(axis=1), a.mean(axis=1), a.std(axis=1))

    def get_outflow(self):
        """
        Get lake outflow data from CSV file.
        """
        if self.xdf is None:
            msg = "First run self.get_data before running this."
            raise TypeError(msg)
        data_dir = os.path.join(os.path.dirname(os.path.abspath(
                                inspect.getfile(inspect.currentframe()))),
                                "data")
        a = np.load(get_data('data/outflow_prior.npz'))
        z, m_out_min, m_out_max, m_out_mean = (a['z'], a['o_min'],
                                               a['o_max'], a['o_mean'])
        f_m_out_min = interp1d(z, m_out_min, fill_value='extrapolate')
        f_m_out_max = interp1d(z, m_out_max, fill_value='extrapolate')
        f_m_out_mean = interp1d(z, m_out_max, fill_value='extrapolate')
       
        new_z = self.xdf.loc[:, 'z', 'val'].data
        m_out = f_m_out_mean(new_z)
        m_err = (f_m_out_min(new_z) + f_m_out_max(new_z))/2.
        
        self.xdf.loc[:, 'm_out', 'val'] = m_out 
        self.xdf.loc[:, 'm_out', 'std'] = m_err

    def get_MetService_wind(self, elev=3000, volcano='Ruapehu'):
        """
        Get wind data from MetService wind model files.
        """
        if self.xdf is None:
            msg = "First run self.get_data before running this."
            raise TypeError(msg)

        baseurl = 'http://vulkan.gns.cri.nz:9876/wind'
        request = baseurl + '?starttime={}&endtime={}&volcano={}&elevation={}'
        request = request.format(self.xdf.dates[0].data, self.xdf.dates[-1].data,
                                 volcano, elev)
        df = pd.read_csv(request, index_col=0, parse_dates=True,
                         header=[0, 1])
        mdf = df.groupby(pd.Grouper(freq='D')).mean()
        sdf = df.groupby(pd.Grouper(freq='D')).std()
        ndf = pd.DataFrame({'W': mdf['pmodel']['speed'],
                            'W_err': sdf['pmodel']['speed']},
                           index=mdf.index)
        ndf.index = ndf.index.tz_convert(None)
        ndf = ndf.reindex(self.xdf.dates.data).interpolate()
        self.xdf.loc[:, 'W', 'val'] = ndf['W']
        self.xdf.loc[:, 'W', 'std'] = ndf['W_err']


class WindData:
    """
    Load wind speed data from a CSV file.
    """

    def __init__(self, csvfile=None, default=4.5):
        if csvfile is None:
            self.csvfile = get_data('data/wind.dat')
        else:
            self.csvfile = csvfile
        # check whether file has already been opened
        try:
            self.csvfile.tell()
            self._buf = csvfile
        except AttributeError:
            self._buf = open(csvfile)

        self._default = default
        self.sr = None
        self.get_data = self.get_data_csv

    def get_data_csv(self, start, end):
        if self.sr is None:
            with self._buf:
                windspeed = []
                dates = []
                while True:
                    l = self._buf.readline()
                    if not l:
                        break
                    # l = l.decode()
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
