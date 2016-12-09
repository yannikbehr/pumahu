"""
Compute energy and mass balance for crater lakes from measurements of water
temperature, wind speed and chemical dilution.
"""

from abc import ABCMeta, abstractmethod
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

    def __init__(self, data):
        self._data = data
        self._index = 0
        self._min = 0
        self._max = 0

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        s = self._data[self._index]
        self._index += 1
        return np.random.uniform(s - self._min, s + self._max)

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

    def __init__(self, data):
        self._data = data
        self._index = 0
        self._std = None

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        s = self._data[self._index]
        self._index += 1
        if self._std is None:
            return s
        return np.random.normal(s, self._std)

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
        pass

    def get_data(self):
        pass


class LakeDataFITS(DataLoader):

    def __init__(self):
        pass

    def get_data(self):
        pass


class WindDataCSV(DataLoader):

    def __init__(self, csvfile):
        pass

    def get_data(self):
        pass


class Clemb:

    def __init__(self, lakedata, winddata):
        pass

    def run(self, nsamples=1):
        iterables = [pd.date_range('1/1/2000', periods=5), range(nsamples)]
        midx = pd.MultiIndex.from_product(iterables)
        df = pd.DataFrame({'steam': np.random.randn(5 * nsamples)}, index=midx)
        return df
