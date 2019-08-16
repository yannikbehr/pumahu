from collections import defaultdict
import inspect
import os
import unittest

import numpy as np
import pandas as pd

from clemb.data import LakeData
from clemb import Clemb 
from clemb.syn_model import SynModel


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

    def test_clemb(self):
        with get_data('data/data.dat') as lb, get_data('data/wind.dat') as wb:
            c = Clemb(LakeDataCSV(lb), WindDataCSV(wb, default=0.0),
                      start='2003-01-16', end='2010-01-29')
            a, vol = c.fullness(pd.Series(np.ones(10) * 2529.4))

            fvol = np.ones(10) * 8880.29883
            diffvol = abs(vol - fvol) / fvol * 100.
            fa = np.ones(10) * 196370.188
            diffa = abs(a - fa) / fa * 100.
            # Probably due to different precisions the numbers between the
            # original Fortran code and the Python code differ slightly
            self.assertTrue(np.all(diffvol < 0.0318))
            self.assertTrue(np.all(diffa < 0.000722))
            loss, ev = c.es(35.0, 5.0, 200000)
            self.assertAlmostEqual(loss, 20.61027, 5)
            self.assertAlmostEqual(ev, 4.87546, 5)
            rs = c.run([0])
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
            diffmass = np.abs(rs['mass'][0] - ts['mass']) / ts['mass'] * 100.
            # Due to above mentioned difference in the volume computation the
            # estimated mass of the crater lake also differs
            self.assertTrue(np.all(diffmass < 0.041))

            # Test the update function
            t1 = c.get_variable('t')
            h1 = c.get_variable('h')
            wd1 = c.get_variable('wind')
            t1.std = 0.5
            c.update_data('2010-01-01', '2010-01-29')
            t2 = c.get_variable('t')
            h2 = c.get_variable('h')
            wd2 = c.get_variable('wind')
            self.assertEqual(t1.std, t2.std)
            self.assertEqual(h1.data['20100128'], h2.data['20100128'])
            self.assertEqual(wd1.data['20100128'], wd2.data['20100128'])

        # Test the update function with the FITS interface
        c1 = Clemb(LakeDataFITS(), WindDataCSV(),
                   start='2003-01-16', end='2010-01-29')
        t1 = c1.get_variable('t')
        h1 = c1.get_variable('h')
        wd1 = c1.get_variable('wind')
        t1.std = 0.5
        c1.update_data('2010-01-01', '2010-01-29')
        t2 = c1.get_variable('t')
        h2 = c1.get_variable('h')
        wd2 = c1.get_variable('wind')
        self.assertEqual(t1.std, t2.std)
        self.assertEqual(h1.data['20100128'], h2.data['20100128'])
        self.assertEqual(wd1.data['20100128'], wd2.data['20100128'])

    def test_dilution(self):
        with get_data('data/data.dat') as lb, get_data('data/wind.dat') as wb:
            c = Clemb(LakeDataCSV(lb), WindDataCSV(wb, default=0.0),
                      start='2003-01-16', end='2010-01-29')
            c.update_data(start='2003-01-16', end='2010-01-29')
            dv = c.get_variable('dv')
            dv.data = pd.Series(np.zeros(dv.data.size), index=dv.data.index)
            c.update_data(start='2003-01-16', end='2003-01-20')
            self.assertTrue(np.all(c.get_variable('dv').data < 1.0))

    def test_with_dq(self):
        s = SynModel()
        df = s.run(1000., mode='test')
        print(df)

    def test_clemb_synthetic(self):
        c = Clemb(None, None, None, None, pre_txt='syn1',
                  resultsd='./data', save_results=False)
        df = SynModel().run(1000., mode='test')
        c._df = df
        c._dates = df.index
        rs = c.run_forward(nsamples=2000, nresample=-1, m_out_max=40.,
                           m_in_max=40., q_in_max=1500., new=True,
                           prior_sampling=False, tolZ=1e-3,
                           prior_resample=10000, Q_scale=300.,
                           dQdT=3e3, tolH=3e30, seed=42)
        np.testing.assert_array_almost_equal(rs['exp'].loc[:, 'q_in'].data,
                                             np.array([201.444228,
                                                       302.327733,
                                                       608.507819]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(rs['var'].loc[:, 'q_in'].data,
                                             np.array([21113.987569,
                                                       38603.58335,
                                                       54191.758414]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(rs['z'].data,
                                             np.array([[-9.533294, 0.170592],
                                                       [-9.724431, 0.171948],
                                                       [-9.191915, 0.156072]]),
                                             decimal=6)


def suite():
    return unittest.makeSuite(ClembTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
