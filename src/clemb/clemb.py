"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import pkg_resources


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
    #ndf = ndf.resample('1D', label='left', loffset='12H').mean()
    ndf = ndf.resample('1D', label='left').mean()
    return ndf


class Variable(metaclass=ABCMeta):
    """
    A base class for a stochastic variable.
    """

    def __init__(self, series):
        self._ser = series
        self._size = len(self._ser)
        self._dates = self._ser.index
        self._index = 0

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    def reset(self):
        self._index = 0

    @property
    def data(self):
        pass

    @data.setter
    @abstractmethod
    def data(self, values):
        pass


class Uniform(Variable):
    """
    Re-define a set of data points as a sample of a uniform distribution.
    """

    def __init__(self, series):
        super().__init__(series)
        self._min = 0
        self._max = 0

    def __next__(self):
        if self._index >= self._size:
            raise StopIteration
        v = self._ser[self._index]
        d = self._ser.index[self._index]
        self._index += 1
        return (d, np.random.uniform(v - self._min, v + self._max))

    def __getitem__(self, datestring):
        v = self._ser.loc[datestring]
        return np.random.uniform(v - self._min, v + self._max)

    def __setitem__(self, datestring, value):
        self._ser.loc[datestring] = value

    @property
    def data(self):
        return pd.Series(np.random.uniform(self._ser - self._min,
                                           self._ser + self._max),
                         index=self._dates)

    @data.setter
    def data(self, series):
        self._ser = series

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self._min = value

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = value


class Gauss(Variable):
    """
    Re-define a set of data points as a sample of a gaussian distribution.
    """

    def __init__(self, series):
        super().__init__(series)
        self._std = None

    def __next__(self):
        if self._index >= self._size:
            raise StopIteration
        v = self._ser[self._index]
        d = self._ser.index[self._index]
        self._index += 1
        if self._std is None:
            return (d, v)
        return (d, np.random.normal(v, self._std))

    def __getitem__(self, datestring):
        v = self._ser.loc[datestring]
        if self._std is None:
            return v
        return np.random.normal(v, self._std)

    def __setitem__(self, datestring, value):
        self._ser[datestring] = value

    @property
    def data(self):
        if self._std is not None:
            return pd.Series(np.random.normal(self._ser, self._std),
                             index=self._dates)
        return self._ser

    @data.setter
    def data(self, series):
        self._ser = series

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value):
        self._std = value


class DataLoader(metaclass=ABCMeta):

    @abstractmethod
    def get_data(self, start, end):
        pass


class LakeDataCSV(DataLoader):
    """
    Load the lake measurements from a CSV file.
    """

    def __init__(self, buf=None):
        if buf is not None:
            self._buf = buf
        else:
            self._buf = pkg_resources.resource_stream(
                __name__, 'data/data.dat')

    def get_data(self, start, end):
        with self._buf:
            rd = defaultdict(list)
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
            df = pd.DataFrame(rd, index=rd['date'])
            df = df.reindex(pd.date_range(start=start, end=end)).interpolate()
            vd = {}
            for _c in df.columns:
                vd[_c] = Gauss(df[_c])
        return vd


class LakeDataFITS(DataLoader):
    """
    Load the lake measurements from the FITS database.
    """

    def __init__(self, url="https://fits.geonet.org.nz/observation"):
        self.base_url = url

    def get_data(self, start, end):
        # Get temperature
        # Temperature has been recorded by 3 different sensors so 3 individual
        # requests have to be made
        url = "{}?siteID=RU001&networkID=VO&typeID=t&methodID={}"
        names = ['t', 't_err']
        tdf1 = pd.read_csv(url.format(self.base_url, 'therm'),
                           index_col=0, names=names, skiprows=1,
                           parse_dates=True)
        tdf2 = pd.read_csv(url.format(self.base_url, 'thermcoup'),
                           index_col=0, names=names, skiprows=1,
                           parse_dates=True)
        tdf3 = pd.read_csv(url.format(self.base_url, 'logic'),
                           index_col=0, names=names, skiprows=1,
                           parse_dates=True)
        tdf3 = tdf3.combine_first(tdf2)
        tdf3 = tdf3.combine_first(tdf1)
        tdf = df_resample(tdf3)

        # Get lake level
        # The lake level data is stored qith respect to the overflow level of
        # the lake. Unfortunately, that level has changed over time so to get
        # the absolute lake level altitude, data from different periods have to
        # be corrected differently. Also, lake level data has been measured by
        # different methods requiring several requests.
        url = "{}?siteID={}&networkID=VO&typeID=z"
        names = ['h', 'h_err']
        ldf = pd.read_csv(url.format(self.base_url, 'RU001'),
                          index_col=0, names=names, skiprows=1,
                          parse_dates=True)
        ldf1 = pd.read_csv(url.format(self.base_url, 'RU001A'),
                           index_col=0, names=names, skiprows=1,
                           parse_dates=True)
        ldf = ldf.combine_first(ldf1)
        ldf.loc[ldf.index < '1997-01-01', 'h'] = 2530. + \
            ldf.loc[ldf.index < '1997-01-01', 'h']
        ldf.loc[(ldf.index > '1997-01-01') & (ldf.index < '2012-12-31'),
                'h'] = 2529.5 + \
            (ldf.loc[(ldf.index > '1997-01-01') &
                     (ldf.index < '2012-12-31'), 'h'] - 1.3)
        ldf.loc[ldf.index > '2016-01-01', 'h'] = 2529.35 + \
            (ldf.loc[ldf.index > '2016-01-01', 'h'] - 2.0)
        ldf = df_resample(ldf)

        # Get Mg++
        url = "{}?siteID=RU001&networkID=VO&typeID=Mg-w"
        names = ['m', 'm_err']
        mdf = pd.read_csv(url.format(self.base_url),
                          index_col=0, names=names, skiprows=1,
                          parse_dates=True)
        mdf = df_resample(mdf)

        # Get Cl-
        url = "{}?siteID=RU001&networkID=VO&typeID=Cl-w"
        names = ['c', 'c_err']
        cdf = pd.read_csv(url.format(self.base_url), index_col=0,
                          names=names, skiprows=1, parse_dates=True)
        cdf = df_resample(cdf)

        df = pd.merge(tdf, ldf, left_index=True, right_index=True, how='outer')
        df = pd.merge(df, mdf, left_index=True, right_index=True, how='outer')
        df = pd.merge(df, cdf, left_index=True, right_index=True, how='outer')
        df = df.fillna(method='pad').dropna()
        start = max(df.index.min(), pd.Timestamp(start))
        end = min(df.index.max(), pd.Timestamp(end))
        df = df.loc[start:end]
        vd = {}
        for c in ['t', 'h', 'm', 'c']:
            vd[c] = Gauss(df[c])
        vd['dv'] = Gauss(pd.Series(np.ones(df.index.size), index=df.index))
        return vd

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
        for i in range(d['t'].data.size):
            date = d['t'].data.index[i]
            s_input = (date.year, date.month, date.day, d['t'].data.ix[i],
                       d['h'].data.ix[i], fl, int(round(d['m'].data.ix[i])),
                       int(round(d['c'].data.ix[i])), dr, o18, deut)
            s_format = "{:<6d}{:<4d}{:<4d}{:<7.1f}{:<8.1f}{:<4d}{:<6d}{:<6d}"
            s_format += "{:<5.1f}{:<7.1f}{:<7.1f}\n"
            line = s_format.format(*s_input)
            lines.append(line)
        with open(filename, 'w') as fh:
            fh.writelines(lines)


class WindDataCSV(DataLoader):
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

    def get_data(self, start, end):
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
                dates.append(np.datetime64('{}-{:02d}-{:02d}'.format(y, m, d)))
                windspeed.append(ws)
            sr = pd.Series(windspeed, index=dates)
            sr = sr.reindex(pd.date_range(start=start, end=end)).fillna(
                self._default)
        return Gauss(sr)


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
        self._ld = lakedata.get_data(start, end)
        self._wd = winddata.get_data(start, end)
        self._dates = self._ld['t'].data.index
        self._enthalpy = Uniform(pd.Series(np.ones(self._dates.size) * 6.0,
                                           index=self._dates))

    def get_variable(self, key):
        if key in self._ld:
            return self._ld[key]
        elif key.lower() == 'wind':
            return self._wd
        elif key.lower() == 'enthalpy':
            return self._enthalpy
        else:
            raise AttributeError('Unknown variable name.')

    def run(self, sampleidx):
        """
        Compute the amount of steam and energy that has to be put into a crater 
        lake to cause an observed temperature change.
        """
        sidx = np.array(sampleidx)
        nsamples = len(sidx)
        if nsamples < 1:
            return None
        ndata = self._dates.size - 1
        results = {}
        keys = ['steam', 'pwr', 'evfl', 'fmelt', 'inf', 'fmg', 'mgt', 'mg',
                'fcl', 'clt', 'cl', 'mass', 't', 'wind']
        for k in keys:
            results[k] = np.zeros(nsamples * ndata)

        for n in range(nsamples):
            df = pd.DataFrame({'t': self._ld['t'].data, 'm': self._ld['m'].data,
                               'c': self._ld['c'].data, 'h': self._ld['h'].data,
                               'dv': self._ld['dv'].data, 'w': self._wd.data},
                              index=self._dates)
            nd = (df.index[1:] - df.index[:-1]).days
            # time interval in Megaseconds
            timem = 0.0864 * (df.index[1:] - df.index[:-1]).days
            density = 1.003 - 0.00033 * df['t']
            a, vol = self.fullness(df['h'])

            # Dilution due to loss of water
            dr = df['dv'][1:].values

            mass = vol[1:] * density[1:].values
            massp = vol[:-1] * density[:-1].values

            # Mass balance for Mg++
            mgt = np.zeros(df['m'].size)
            drmg = np.ones(df['m'].size - 1)
            mgt[0] = massp[0] * df['m'][0]
            mgt[1:] = mass * df['m'][1:].values - \
                massp * df['m'][:-1].values / dr
            mgt = mgt.cumsum()
            fmg = np.diff(mgt) / nd
            idx = np.where(
                np.diff(mgt[::-1])[::-1] > 0.02 * mass * df['m'][1:].values)
            drmg[idx] = 0.98 * massp[idx] * df['m'][:-1].values[idx] / \
                (mass[idx] * df['m'][1:].values[idx])

            # Mass balance for Cl-
            clt = np.zeros(df['c'].size)
            drcl = np.ones(df['c'].size - 1)
            clt[0] = massp[0] * df['c'][0]
            clt[1:] = mass * df['c'][1:].values - \
                massp * df['c'][:-1].values / dr
            clt = clt.cumsum()
            fcl = np.diff(clt) / nd
            idx = np.where(
                np.diff(clt[::-1])[::-1] > 0.02 * mass * df['c'][1:].values)
            drcl[idx] = 0.98 * massp[idx] * df['c'][:-1].values[idx] / \
                (mass[idx] * df['c'][1:].values[idx])

            # Net mass input to lake [kT]
            inf = massp * (dr - 1.0)  # input to replace outflow
            inf = inf + mass - massp  # input to change total mass
            loss, ev = self.es(df['t'][1:].values, df['w'][1:].values, a[1:])

            # Energy balances [TJ];
            # es = Surface heat loss, el = change in stored energy
            e = loss + \
                self.el(df['t'][:-1].values, df['t'][1:].values, vol[1:])

            # e is energy required from steam, so is reduced by sun energy
            e -= self.esol(df.index[:-1], df.index[1:], a[1:])
            # Energy = Mass * Enthalpy
            steam = e / (self._enthalpy.data[1:].values -
                         0.004 * df['t'][1:].values)
            evap = ev  # Evaporation loss
            meltf = inf + evap - steam  # Conservation of mass

            # Correction for energy to heat incoming meltwater
            # FACTOR is ratio: Mass of steam/Mass of meltwater (0 degrees
            # C)
            factor = df['t'][1:].values * 0.004 / \
                (self._enthalpy.data[1:].values - df['t'][1:].values * 0.004)
            meltf = meltf / (1.0 + factor)  # Therefore less meltwater
            steam = steam + meltf * factor  # ...and more steam
            # Correct energy input also
            e += meltf * df['t'][1:].values * 0.004

            # Flows are total amounts/day
            id0 = n * ndata
            id1 = (n + 1) * ndata
            results['steam'][id0:id1] = steam / nd  # kT/day
            results['pwr'][id0:id1] = e / timem  # MW
            results['evfl'][id0:id1] = evap / nd  # kT/day
            results['fmelt'][id0:id1] = meltf / nd  # kT/day
            results['mass'][id0:id1] = mass
            results['inf'][id0:id1] = inf
            results['t'][id0:id1] = df['t'][1:].values
            results['fmg'][id0:id1] = fmg
            results['mgt'][id0:id1] = mgt[1:]
            results['mg'][id0:id1] = df['m'][1:].values
            results['fcl'][id0:id1] = fcl
            results['clt'][id0:id1] = clt[1:]
            results['cl'][id0:id1] = df['c'][1:].values
            results['wind'][id0:id1] = df['w'][1:].values

        iterables = [sidx, df.index[1:]]
        midx = pd.MultiIndex.from_product(iterables)
        df = pd.DataFrame(results, index=midx)
        return df

    def fullness(self, hgt):
        """
        Calculate volume and area from lake level.
        """
        h = hgt.copy()
        a = np.zeros(h.size)
        vol = np.zeros(h.size)
        idx1 = np.where(h < 2400.)
        idx2 = np.where(h >= 2400.)
        h.iloc[idx1] = 2529.4
        vol[idx1] = (4.747475 * np.power(h.iloc[idx1], 3) -
                     34533.8 * np.power(h.iloc[idx1], 2) + 83773360. *
                     h.iloc[idx1] - 67772125000.) / 1000.
        a[idx1] = 193400
        # Calculate from absolute level
        vol[idx2] = 4.747475 * np.power(h.iloc[idx2], 3) - \
            34533.8 * np.power(h.iloc[idx2], 2) + 83773360. * h.iloc[idx2] - \
            67772125000.
        h.iloc[idx2] += 1.0
        v1 = 4.747475 * np.power(h.iloc[idx2], 3) - \
            34533.8 * np.power(h.iloc[idx2], 2) + 83773360. * h.iloc[idx2] - \
            67772125000.
        a[idx2] = v1 - vol[idx2]
        vol[idx2] /= 1000.
        return (a, vol)

    def esol(self, d1, d2, a):
        """
        Solar Incident Radiation Based on yearly guess & month.
        """
        return (d2 - d1).days * a * 0.000015 * \
            (1 + 0.5 * np.cos(((d2.month - 1) * 3.14) / 6.0))

    def el(self, t1, t2, vol):
        """
        Change in Energy stored in the lake [TJ]
        """
        return (t2 - t1) * vol * 0.0042

    def es(self, t, w, a):
        """
        Energy loss from lake surface in TJ/day. Equations apply for nominal
        surface area of 200000 square metres. Since area is only to a small
        power, this error is negligible.
        """

        l = 500  # Characteristic length of lake
        # Expressions for H2O properties as function of temperature
        # Vapour Pressure Function from CIMO Guide (WMO, 2008)
        vp = 6.112 * np.exp(17.62 * t / (243.12 + t))
        # t - 1 for surface temperature
        vp1 = 6.112 * np.exp(17.62 * (t - 1.) / (242.12 + t))
        # Vapour Density from Hyperphysics Site
        vd = .006335 + .0006718 * t - .000020887 * \
            t * t + .00000073095 * t * t * t

        # First term is for radiation, Power(W) = aC(Tw^4 - Ta^4)A
        # Temperatures absolute, a is emissivity, C Stefans Constant
        tk = t + 273.15
        tl = 0.9 + 273.15  # 0.9 C is air temperature
        er = 0.8 * 5.67E-8 * a * (tk**4 - tl**4)

        # Free convection formula from Adams et al(1990)
        # Power (W) = A * factor * delT^1/3 * (es-ea)
        # where factor = 2.3 at 25C and 18% less at 67 C, hence
        # factor = 2.55 - 0.01 * Twr.
        # For both delT and es, we make a 1 C correction, for surface temp
        # below bulk water temp. SVP at average air tmperature 6.5 mBar

        efree = a * (2.55 - 0.01 * t) * (t - 1.9) ** (1 / 3.0) * (vp1 - 6.5)

        # Forced convection by Satori's Equation
        # Evaporation (kg/s/m2) =
        # (0.00407 * W**0.8 / L**0.2 - 0.01107/L)(Pw-Pd)/P
        # Latent heat of vapourization about 2400 kJ/kg in range 20 - 60 C,
        # Atmospheric Pressure 750 mBar at Crater Lake

        eforced = a * (0.00407 * w**0.8 / l**0.2 - 0.01107 / l) * \
            (vp - 6.5) / 800. * 2400000  # W
        ee = np.sqrt(efree**2 + eforced**2)

        # The ratio of Heat Loss by Convection to that by Evaporation is
        # rhoCp/L * (Tw - Ta)/(qw - qa) #rho is air density .948 kg/m3, Cp
        # Specific Heat of Air 1005 J/kg degC, qw & qa are Sat Vap Density

        ratio = .948 * (1005 / 2400000.) * (t - 0.9) / (vd - .0022)

        # The power calculation is in W. Calculate Energy Loss (TW/day) and
        # evaporative volume loss in kT/day
        ev = 86400 * ee / 2.4e12  # kT/day
        loss = (er + ee * (1 + ratio)) * 86400 * 1.0e-12  # TW/day

        return (loss, ev)
