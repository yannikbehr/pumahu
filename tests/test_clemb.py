from collections import defaultdict
import inspect
import os
import unittest

import numpy as np
import pandas as pd

from clemb import LakeDataCSV, LakeDataFITS, WindDataCSV, Clemb


class ClembTestCase(unittest.TestCase):

    def load_test_results(self):
        with open(os.path.join(self.data_dir, 'LAKEOUTm.DAT')) as f:
            # skip header lines
            for i in range(4):
                f.readline()
            # now read until the next empty line
            data = defaultdict(list)
            dates = []
            while True:
                l = f.readline()
                if l == '\n':
                    break
                yr, mon, day, temp, stfl, pwr, evfl, fmelt, inf, drm, mmp,\
                    fmg, fcl, o18, o18m, o18f, h2, h22, m = l.split()
                dates.append(
                    np.datetime64('{}-{:02d}-{:02d}'.format(
                        yr, int(mon), int(day))))
                data['t'].append(float(temp))
                data['steam'].append(float(stfl))
                data['pwr'].append(float(pwr))
                data['evfl'].append(float(evfl))
                data['fmelt'].append(float(fmelt))
                data['inf'].append(float(inf))
                data['fmg'].append(float(fmg))
                data['fcl'].append(float(fcl))
                data['mass'].append(float(m))

        return pd.DataFrame(data, index=dates)

    def load_input(self):
        with open(os.path.join(self.data_dir, 'input.dat')) as f:
            # skip header line
            f.readline()

            data = defaultdict(list)
            while True:
                l = f.readline()
                if l == '\n':
                    break
                if not l:
                    break
                try:
                    yr, mo, dy, t, h, fl, m, c, dv, nd, w, o18, h2 = l.split()
                except:
                    print(l)
                    raise
                data['date'].append(np.datetime64('{}-{:02d}-{:02d}'.format(
                    yr, int(mo), int(dy))))
                data['temp'].append(float(t))
                data['hgt'].append(float(h))
                data['mg'].append(float(m))
                data['cl'].append(float(c))
                data['wind'].append(float(w))
                data['o18'].append(float(o18))
                data['h2'].append(float(h2))
                data['nd'].append(int(nd))
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
        vd = dl.get_data()
        ti = self.load_input()
        temp = [t for d, t in vd['t']]
        hgt = [h for d, h in vd['h']]
        mg = [m for d, m in vd['m']]
        cl = [c for d, c in vd['c']]
        o18 = [o for d, o in vd['o18']]
        h2 = [h for d, h in vd['h2']]
        dno = [d for dt, d in vd['nd']]
        dt = [dt for dt, d in vd['date']]
        np.testing.assert_array_almost_equal(temp, ti['temp'], 1)
        np.testing.assert_array_equal(dno, ti['nd'])
        np.testing.assert_array_almost_equal(hgt, ti['hgt'], 2)
        np.testing.assert_array_almost_equal(mg, ti['mg'], 3)
        np.testing.assert_array_almost_equal(cl, ti['cl'], 3)
        np.testing.assert_array_almost_equal(o18, ti['o18'], 2)
        np.testing.assert_array_almost_equal(h2, ti['h2'], 2)
        np.testing.assert_array_equal(
            np.array(dt, dtype='datetime64[ns]'), ti['date'])

    def test_wind_data_csv(self):
        dl = WindDataCSV(os.path.join(self.data_dir, 'wind.dat'))
        df = dl.get_data()
        ti = self.load_input()
        ws = []
        for dt in ti['date']:
            try:
                ws.append(df[dt])
            except KeyError:
                ws.append(0.0)
        np.testing.assert_array_almost_equal(ws, ti['wind'], 1)

    def test_clemb(self):
        ldata = os.path.join(self.data_dir, 'data.dat')
        wdata = os.path.join(self.data_dir, 'wind.dat')
        c = Clemb(LakeDataCSV(ldata), WindDataCSV(wdata))
        a, vol = c.fullness(2529.4)
        fvol = 8880.29883
        diffvol = abs(vol - fvol) / fvol * 100.
        fa = 196370.188
        diffa = abs(a - fa) / fa * 100.
        # Probably due to different precisions the numbers between the
        # original Fortran code and the Python code differ slightly
        self.assertTrue(diffvol < 0.0318)
        self.assertTrue(diffa < 0.000722)
        loss, ev = c.es(35.0, 5.0, 200000)
        self.assertAlmostEqual(loss, 19.913621, 5)
        self.assertAlmostEqual(ev, 5.119750, 5)
        rs = c.run()
        ts = self.load_test_results()
        np.testing.assert_array_almost_equal(rs['steam'],
                                             ts['steam'], 1)
        np.testing.assert_array_almost_equal(rs['pwr'],
                                             ts['pwr'], 1)
        np.testing.assert_array_almost_equal(rs['evfl'],
                                             ts['evfl'], 1)
        np.testing.assert_array_almost_equal(rs['fmelt'],
                                             ts['fmelt'], 1)
        np.testing.assert_array_almost_equal(rs['inf'],
                                             ts['inf'], 1)
        np.testing.assert_array_almost_equal(rs['fmg'],
                                             ts['fmg'], 0)
        np.testing.assert_array_almost_equal(rs['fcl'],
                                             ts['fcl'], 0)
        diffmass = np.abs(rs['mass'][:, 0] - ts['mass']) / ts['mass'] * 100.
        # Due to above mentioned difference in the volume computation the
        # estimated mass of the crater lake also differs
        self.assertTrue(np.all(diffmass < 0.041))


def suite():
    return unittest.makeSuite(ClembTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')