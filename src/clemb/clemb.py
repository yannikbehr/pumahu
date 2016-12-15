"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import exp, cos, sqrt

import numpy as np
import pandas as pd


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
        return pd.Series(np.random.uniform(self._ser - 2.0, self._ser + 2.0),
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

    def __init__(self, csvfile):
        self._fin = csvfile

    def get_data(self, start, end):
        rd = defaultdict(list)
        t0 = np.datetime64('2000-01-01')
        with open(self._fin) as f:
            while True:
                l = f.readline()
                if not l:
                    break
                # ignore commented lines
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

    def __init__(self):
        pass

    def get_data(self):
        pass


class WindDataCSV(DataLoader):
    """
    Load wind speed data from a CSV file.
    """

    def __init__(self, csvfile):
        self._fin = csvfile

    def get_data(self, start, end, default=0.0):
        windspeed = []
        dates = []
        with open(self._fin) as f:
            while True:
                l = f.readline()
                if not l:
                    break
                a = l.split()
                y, m, d = map(int, a[0:3])
                ws, wd = map(float, a[3:])
                dates.append(np.datetime64('{}-{:02d}-{:02d}'.format(y, m, d)))
                windspeed.append(ws)
        sr = pd.Series(windspeed, index=dates)
        sr = sr.reindex(pd.date_range(start=start, end=end)).fillna(default)
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
        self._dates = self._ld['date'].data.index
        self._enthalpy = Uniform(pd.Series(np.ones(self._dates.size) * 6.0,
                                           index=self._dates))

    def run(self, nsamples=1):
        """
        Compute the amount of steam and energy that has to be put into a crater 
        lake to cause an observed temperature change.
        """
        mgt = Gauss(pd.Series(np.zeros(self._dates.size), index=self._dates))
        fmg = Gauss(pd.Series(np.zeros(self._dates.size), index=self._dates))
        clt = Gauss(pd.Series(np.zeros(self._dates.size), index=self._dates))
        fcl = Gauss(pd.Series(np.zeros(self._dates.size), index=self._dates))
        m = self._ld['m']
        c = self._ld['c']
        t = self._ld['t']
        h = self._ld['h']
        dv = self._ld['dv']
        w = self._wd

        d0, _ = next(self._ld['date'])
        a, vol = self.fullness(h[d0])
        density = 1.003 - 0.00033 * t[d0]
        mass = vol * density
        mgt[d0] = m[d0] * mass
        clt[d0] = c[d0] * mass
        dold = d0
        dates = []
        results = {'steam': [], 'pwr': [], 'evfl': [], 'fmelt': [],
                   'inf': [], 'fmg': [], 'fcl': [], 'mass': []}
        for dt, _ in self._ld['date']:
            for _n in range(nsamples):
                # time interval in Megaseconds
                timem = 0.0864 * (dt - dold).days
                density = 1.003 - 0.00033 * t[dt]
                massp = mass
                a, vol = self.fullness(h[dt])
                mass = vol * density
                dr = dv[dt]
                mgt[dt] = mgt[dold] + mass * m[dt] - massp * m[dold] / dr
                fmg[dt] = (mgt[dt] - mgt[dold]) / (dt - dold).days
                if (mgt[dold] - mgt[dt]) > 0.02 * mass * m[dt]:
                    drmg = 0.98 * massp * m[dold] / mass * m[dt]

                clt[dt] = clt[dold] + mass * c[dt] - massp * c[dold] / dr
                fcl[dt] = (clt[dt] - clt[dold]) / (dt - dold).days
                if (fcl[dold] - fcl[dt]) > 0.02 * mass * c[dt]:
                    drmg = 0.98 * massp * c[dold] / mass * c[dt]

                # Net mass input to lake [kT]
                inf = massp * (dr - 1.0)  # input to replace outflow
                inf = inf + mass - massp  # input to change total mass
                try:
                    loss, ev = self.es(t[dt], w[dt], a)
                except KeyError:
                    # No wind data
                    loss, ev = self.es(t[dt], 0.0, a)

                # Energy balances [TJ];
                # es = Surface heat loss, el = change in stored energy
                e = loss + self.el(t[dold], t[dt], vol)

                # e is energy required from steam, so is reduced by sun energy
                e -= self.esol(dold, dt, a)
                # Energy = Mass * Enthalpy
                steam = e / (self._enthalpy[dt] - 0.004 * t[dt])
                evap = ev  # Evaporation loss
                meltf = inf + evap - steam  # Conservation of mass

                # Correction for energy to heat incoming meltwater
                # FACTOR is ratio: Mass of steam/Mass of meltwater (0 degrees
                # C)
                factor = t[dt] * 0.004 / (self._enthalpy[dt] - t[dt] * 0.004)
                meltf = meltf / (1.0 + factor)  # Therefore less meltwater
                steam = steam + meltf * factor  # ...and more steam
                e += meltf * t[dt] * 0.004  # Correct energy input also

                # Flows are total amounts/day
                dates.append(dt)
                results['steam'].append(steam / (dt - dold).days)  # kT/day
                results['pwr'].append(e / timem)  # MW
                results['evfl'].append(evap / (dt - dold).days)  # kT/day
                results['fmelt'].append(meltf / (dt - dold).days)  # kT/day
                results['mass'].append(mass)
                results['inf'].append(inf)
                results['fmg'].append(fmg[dt])
                results['fcl'].append(fcl[dt])
                dold = dt

        iterables = [dates, range(nsamples)]
        midx = pd.MultiIndex.from_product(iterables)
        df = pd.DataFrame(results, index=midx)
        return df

    def fullness(self, hgt):
        """
        Calculate volume and area from lake level.
        """
        if hgt < 2400.:
            h = 2529.4
            vol = (4.747475 * h * h * h - 34533.8 * h * h + 83773360. * h - 67772125000.) / \
                1000.
            a = 193400
        else:
            # Calculate from absolute level
            h = hgt
            v = 4.747475 * h * h * h - 34533.8 * h * h + 83773360. * h - \
                67772125000.
            h = h + 1.0
            v1 = 4.747475 * h * h * h - 34533.8 * h * h + 83773360. * h - \
                67772125000.
            a = v1 - v
            vol = v / 1000.
        return (a, vol)

    def esol(self, d1, d2, a):
        """
        Solar Incident Radiation Based on yearly guess & month.
        """
        return (d2 - d1).days * a * 0.000015 * \
            (1 + 0.5 * cos(((d2.month - 1) * 3.14) / 6.0))

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
        vp = 6.112 * exp(17.62 * t / (243.12 + t))
        # t - 1 for surface temperature
        vp1 = 6.112 * exp(17.62 * (t - 1.) / (242.12 + t))
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
        ee = sqrt(efree**2 + eforced**2)

        # The ratio of Heat Loss by Convection to that by Evaporation is
        # rhoCp/L * (Tw - Ta)/(qw - qa) #rho is air density .948 kg/m3, Cp
        # Specific Heat of Air 1005 J/kg degC, qw & qa are Sat Vap Density

        ratio = .948 * (1005 / 2400000.) * (t - 0.9) / (vd - .0022)

        # The power calculation is in W. Calculate Energy Loss (TW/day) and
        # evaporative volume loss in kT/day
        ev = 86400 * ee / 2.4e12  # kT/day
        loss = (er + ee * (1 + ratio)) * 86400 * 1.0e-12  # TW/day

        return (loss, ev)
