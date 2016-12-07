import inspect
import os
import unittest

import numpy as np
import pandas as pd

from clemb import LakeDataCSV, LakeDataFITS, WindDataCSV, ClemB


class ClemTestCase(unittest.TestCase):

    def load_test_results(self):
        with open(os.path.join(self.data_dir, 'LAKEOUTm.DAT')) as f:
            # skip header lines
            for i in range(4):
                f.readline()
            # now read until the next empty line
            data = {'date': [], 't': [], 'm': [], 'steam': []}
            while True:
                l = f.readline()
                if l == '\n':
                    break
                yr, mon, day, temp, stfl, pwr, evfl, fmelt, inf, drm, mmp,\
                    fmg, fcl, o18, o18m, o18f, h2, h22, m = l.split()
                data['date'].append(
                    np.datetime64('{}-{:02d}-{:02d}'.format(
                        yr, int(mon), int(day))))
                data['t'].append(float(temp))
                data['m'].append(float(m))
                data['steam'].append(float(stfl))
        return pd.DataFrame(data)

    def load_input(self):
        with open(os.path.join(self.data_dir, 'input.dat')) as f:
            # skip header line
            f.readline()

            data = {'date': [], 'temp': [], 'hgt': [], 'mg': [], 'cl': [],
                    'wind': [], 'o18': [], 'h2': []}
            while True:
                l = f.readline()
                if l == '\n':
                    break
                yr, mo, dy, t, h, f, m, c, dv, nd, w, o18, h2 = l.split()
                data['date'].append(
                    np.datetime64('{}-{:02d}-{:02d}'.format(
                        yr, int(mo), int(dy))))
                data['temp'].append(float(t))
                data['hgt'].append(float(h))
                data['mg'].append(float(m))
                data['cl'].append(float(c))
                data['wind'].append(float(w))
                data['o18'].append(float(o18))
                data['h2'].append(float(h2))
        return pd.DataFrame(data)

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_lake_data_fits(self):
        dl = LakeDataFITS()
        df = dl.get_data()
        ti = self.load_input()
        np.testing.assert_array_almost_equal(df['temp'],
                                             ti['temp'], 5)

    def test_lake_data_csv(self):
        dl = LakeDataCSV(os.path.join(self.data_dir, 'data.dat'))
        df = dl.get_data()
        ti = self.load_input()
        np.testing.assert_array_almost_equal(df['temp'],
                                             ti['temp'], 5)

    def test_wind_data_csv(self):
        dl = WindDataCSV(os.path.join(self.data_dir, 'wind.dat'))
        df = dl.get_data()
        ti = self.load_input()
        np.testing.assert_array_almost_equal(df['wind'],
                                             ti['wind'], 5)

    def test_clem(self):
        ldata = os.path.join(self.data_dir, 'data.dat')
        wdata = os.path.join(self.data_dir, 'wind.dat')
        c = ClemB(LakeDataCSV(ldata), WindDataCSV(wdata))
        rs = c.run()
        ts = self.load_test_results()
        np.testing.assert_array_almost_equal(rs['steam'],
                                             ts['steam'], 5)


def suite():
    return unittest.makeSuite(ClemTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
