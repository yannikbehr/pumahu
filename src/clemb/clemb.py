"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd


class Variable(metaclass=ABCMeta):

    def __iter__(self):
        return self

    @abstractmethod
    def __init__(self, data):
        pass

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

    def __init__(self, dates, data, name):
        self._df = pd.DataFrame({name: data}, index=dates)
        self._size = len(self._df)
        self._index = 0
        self._min = 0
        self._max = 0

    def __next__(self):
        if self._index >= self._size:
            raise StopIteration
        s = self._df.iloc[self._index]
        self._index += 1
        return (s.name, np.random.uniform(s.values[0] - self._min,
                                          s.values[0] + self._max))

    def __getitem__(self, datestring):
        s = self._df.loc[datestring]
        return (s.name, np.random.uniform(s.values[0] - self._min,
                                          s.values[0] + self._max))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):
        self._data = values

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

    def __init__(self, dates, data, name):
        self._df = pd.DataFrame({name: data}, index=dates)
        self._size = len(self._df)
        self._index = 0
        self._std = None

    def __next__(self):
        if self._index >= self._size:
            raise StopIteration
        s = self._df.iloc[self._index]
        self._index += 1
        if self._std is None:
            return (s.name, s.values[0])
        return (s.name, np.random.normal(s.values[0], self._std))

    def __getitem__(self, datestring):
        s = self._df.loc[datestring]
        if self._std is None:
            return (s.name, s.values[0])
        return (s.name, np.random.normal(s.values[0], self._std))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):
        self._data = values

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value):
        self._std = value


class DataLoader(metaclass=ABCMeta):

    @abstractmethod
    def get_data(self):
        pass


class LakeDataCSV(DataLoader):

    def __init__(self, csvfile):
        self._fin = csvfile

    def get_data(self):
        rd = defaultdict(list)
        rd_old = {}
        n = 0
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
                if n < 1:
                    nstart = no
                    nprev = no
                    rd['nd'].append(no)
                    rd['date'].append(dt)
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
                nfinish = no
                if dr < 0.1:
                    dr = 1.0
                for nn in range(nprev + 1, nfinish + 1):
                    fact = (no - nn) / (no - nprev)
                    rd['t'].append(te + (rd_old['t'] - te) * fact)
                    rd['o18'].append(oheavy + (rd_old['o18'] - oheavy) * fact)
                    rd['h2'].append(deut + (rd_old['h2'] - deut) * fact)
                    rd['m'].append(
                        img / 1000. + (rd_old['m'] - img / 1000.) * fact)
                    rd['c'].append(
                        icl / 1000. + (rd_old['c'] - icl / 1000.) * fact)
                    rd['dv'].append(1.0 + (dr - 1.0) / (no - nprev))
                    rd['nd'].append(nn)
                    rd['date'].append(t0 + (nn + 1) * np.timedelta64(1, 'D'))
                    rd['h'].append(hgt + (rd_old['h'] - hgt) * fact)
                    rd['f'].append(rd_old['f'])
                rd['f'][-1] = fl
                for _k in rd:
                    rd_old[_k] = rd[_k][-1]
                nprev = no
                n += 1
        vd = {}
        for _k in rd:
            vd[_k] = Gauss(rd['date'], rd[_k], _k)
        return vd


class LakeDataFITS(DataLoader):

    def __init__(self):
        pass

    def get_data(self):
        pass


class WindDataCSV(DataLoader):

    def __init__(self, csvfile):
        self._fin = csvfile

    def get_data(self):
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
        return Gauss(dates, windspeed, 'wind')


class Clemb:

    def __init__(self, lakedata, winddata):
        pass

    def run(self, nsamples=1):
        iterables = [pd.date_range('1/1/2000', periods=5), range(nsamples)]
        midx = pd.MultiIndex.from_product(iterables)
        df = pd.DataFrame({'steam': np.random.randn(5 * nsamples)}, index=midx)
        return df
