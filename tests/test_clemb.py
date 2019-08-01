from collections import defaultdict
import inspect
import os
import time
import unittest

import numpy as np
import pandas as pd

from clemb import (LakeDataCSV, LakeDataFITS, WindDataCSV,
                   Clemb, get_data, get_T, get_Mg, get_ll,
                   FITS_request)
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

    def load_fits_input(self):
        with open(os.path.join(self.data_dir, 'fits_table.txt')) as f:
            data = defaultdict(list)
            while True:
                l = f.readline()
                if l == '\n':
                    break
                if not l:
                    break
                try:
                    yr, mo, dy, t, h, fl, m, c, dv, o18, h2 = l.split()
                except:
                    print(l)
                    raise
                data['date'].append(np.datetime64('{}-{:02d}-{:02d}'.format(
                    yr, int(mo), int(dy))))
                data['temp'].append(float(t))
                data['hgt'].append(float(h))
                data['mg'].append(float(m))
                data['cl'].append(float(c))
                data['o18'].append(float(o18))
                data['h2'].append(float(h2))
        return pd.DataFrame(data)

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_FITS_request(self):
        self.assertAlmostEqual(FITS_request('Mg').iloc[0]['obs'],
                               3050.0)
        self.assertAlmostEqual(FITS_request('L').iloc[0]['obs'],
                               2503.0)
        self.assertAlmostEqual(FITS_request('T').iloc[0]['obs'],
                               24.4)

    def test_get_T(self):
        """
        Test different ways of receiving and smoothing temperature readings.
        """
        index = pd.date_range('2019-01-01', '2019-01-03')
        kf_test_frame = pd.DataFrame({'t': np.array([31.93833333,
                                                     31.84910086,
                                                     31.78383591]),
                                      't_err': np.array([0.3, 0.08860859,
                                                         0.24334404]),
                                      't_orig': np.array([31.93833333,
                                                          31.82958333,
                                                          31.98681818])},
                                     index=index)
        dv_test_frame = pd.DataFrame({'t': np.array([31.93833333,
                                                     31.82958333,
                                                     31.98681818]),
                                      't_err': np.array([0.35497295,
                                                         0.66342641,
                                                         0.3821915]),
                                      't_orig': np.array([31.93833333,
                                                          31.82958333,
                                                          31.98681818])},
                                     index=index)
        dkf = get_T(tstart='2019-01-01', tend='2019-01-03', smoothing='kf')
        ddv = get_T(tstart='2019-01-01', tend='2019-01-03', smoothing='dv')
        pd.testing.assert_frame_equal(kf_test_frame, dkf)
        pd.testing.assert_frame_equal(dv_test_frame, ddv)

    def test_get_Mg(self):
        """
        Test different ways of receiving and smoothing temperature readings.
        """
        df = get_Mg(tend='2019-07-31', smoothing='dv')
        nans = np.where(df['Mg'].isna().values, 1, 0).sum()
        no_nans = np.where(df['Mg'].isna().values, 0, 1).sum()
        self.assertEqual(no_nans, 137)
        self.assertEqual(nans, 7670)
        dkf = get_Mg(tstart='2019-01-01', tend='2019-01-03', smoothing='kf')
        ddv = get_Mg(tstart='2019-01-01', tend='2019-01-03', smoothing='dv')
        self.assertAlmostEqual(ddv.iloc[0]['Mg'], 373.0)
        self.assertAlmostEqual(dkf.iloc[0]['Mg'], 373.0)
        self.assertAlmostEqual(dkf.iloc[0]['Mg_err'], 50.0)

    def test_get_ll(self):
        index = pd.date_range('2019-01-01', '2019-01-03')
        kf_test_frame = pd.DataFrame({'h': np.array([2529.32191667,
                                                     2529.33634745,
                                                     2529.33609614]),
                                      'h_err': np.array([0.03,
                                                         0.013484,
                                                         0.01401298]),
                                      'h_orig': np.array([2529.32191667,
                                                          2529.343125,
                                                          2529.33509091])},
                                     index=index)
        dv_test_frame = pd.DataFrame({'h': np.array([2529.32191667,
                                                     2529.343125,
                                                     2529.33509091]),
                                      'h_err': np.array([0.00672385,
                                                         0.01236163,
                                                         0.01548243]),
                                      'h_orig': np.array([2529.32191667,
                                                          2529.343125,
                                                          2529.33509091])},
                                     index=index)
        dkf = get_ll(tstart='2019-01-01', tend='2019-01-03', smoothing='kf')
        ddv = get_ll(tstart='2019-01-01', tend='2019-01-03', smoothing='dv')
        pd.testing.assert_frame_equal(kf_test_frame, dkf)
        pd.testing.assert_frame_equal(dv_test_frame, ddv)

    def test_lake_data_fits(self):
        dl = LakeDataFITS()
        tic = time.time()
        df = dl.get_data('20160603', '20161231')
        toc = time.time()
        td1 = toc - tic
        ti = self.load_fits_input()
        np.testing.assert_array_almost_equal(df['T'].data,
                                             ti['temp'], 1)
        np.testing.assert_array_almost_equal(df['z'].data,
                                             ti['hgt'], 1)
        np.testing.assert_array_almost_equal(df['Mg'].data,
                                             ti['mg'], 0)
        # Make sure that a second request gets the data from the cache
        # instead of requesting it again
        tic = time.time()
        df1 = dl.get_data('20160603', '20161231')
        toc = time.time()
        td2 = toc - tic
        self.assertTrue(td2 / td1 * 100. < 0.1)
        np.testing.assert_array_almost_equal(df1['T'].data,
                                             ti['temp'], 1)
        np.testing.assert_array_almost_equal(df1['z'].data,
                                             ti['hgt'], 1)
        np.testing.assert_array_almost_equal(df1['Mg'].data,
                                             ti['mg'], 0)

    def test_lake_data_csv(self):
        ti = self.load_input()
        with get_data('data/data.dat') as lb:
            dl = LakeDataCSV(lb)
            vd = dl.get_data(start='2003-01-16', end='2010-01-29')
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

        dl1 = LakeDataCSV()
        vd = dl1.get_data(start='2003-01-16', end='2010-01-29')
        np.testing.assert_array_almost_equal(vd['t'].data, ti['temp'], 1)
        np.testing.assert_array_equal(vd['nd'].data, ti['nd'])
        np.testing.assert_array_almost_equal(vd['h'].data, ti['hgt'], 2)
        np.testing.assert_array_almost_equal(vd['m'].data, ti['mg'], 3)
        np.testing.assert_array_almost_equal(vd['c'].data, ti['cl'], 3)
        np.testing.assert_array_almost_equal(vd['o18'].data, ti['o18'], 2)
        np.testing.assert_array_almost_equal(vd['h2'].data, ti['h2'], 2)
        np.testing.assert_array_equal(
            np.array(vd['date'].data.index, dtype='datetime64[ns]'), ti['date'])

    def test_wind_data_csv(self):
        ti = self.load_input()
        with get_data('data/wind.dat') as wb:
            dl = WindDataCSV(wb, default=0.0)
            df = dl.get_data(start='2003-01-16', end='2010-01-29')
            ws = [w for d, w in df]
            np.testing.assert_array_almost_equal(ws, ti['wind'], 1)

        dl1 = WindDataCSV(default=0.0)
        df1 = dl1.get_data(start='2003-01-16', end='2010-01-29')
        ws = [w for d, w in df1]
        np.testing.assert_array_almost_equal(ws, ti['wind'], 1)

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
        df = SynModel().run(1000., nsteps=21)
        df = df[(df.index >= '2017-01-03') & (df.index <= '2017-01-05')]
        c._df = df
        c._dates = df.index
        rs = c.run_forward(nsamples=2000, nresample=-1, m_out_max=40.,
                           m_in_max=40., q_in_max=1500., new=True,
                           prior_sampling=False, tolZ=1e-3,
                           prior_resample=10000, Q_scale=300.,
                           dQdT=3e3, tolH=3e30, seed=42)
        np.testing.assert_array_almost_equal(rs['exp'].loc[:, 'q_in'].data,
                                             np.array([189.551592,
                                                       314.575052]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(rs['var'].loc[:, 'q_in'].data,
                                             np.array([24437.03193914,
                                                       38166.63362941]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(rs['z'].data,
                                             np.array([[-9.968557,  0.175137],
                                                       [-9.567256,  0.164118]]),
                                             decimal=6)




def suite():
    return unittest.makeSuite(ClembTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
