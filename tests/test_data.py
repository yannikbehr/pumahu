from collections import defaultdict
from datetime import timezone
import inspect
import os
import unittest
import warnings

import numpy as np
import pandas as pd

from clemb import get_data
from clemb.data import LakeData, WindData


class DataTestCase(unittest.TestCase):

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

    def test_FITS_request(self):
        ld = LakeData()
        self.assertAlmostEqual(ld.FITS_request('Mg').iloc[0]['obs'],
                               3050.0)
        self.assertAlmostEqual(ld.FITS_request('L').iloc[0]['obs'],
                               2503.0)
        self.assertAlmostEqual(ld.FITS_request('T').iloc[0]['obs'],
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
        ld = LakeData()
        dkf = ld.get_T(tstart='2019-01-01', tend='2019-01-03', smoothing='kf')
        ddv = ld.get_T(tstart='2019-01-01', tend='2019-01-03', smoothing='dv')
        pd.testing.assert_frame_equal(kf_test_frame, dkf)
        pd.testing.assert_frame_equal(dv_test_frame, ddv)

    def test_get_Mg(self):
        """
        Test different ways of receiving and smoothing temperature readings.
        """
        ld = LakeData()
        df = ld.get_Mg(tend='2019-07-31', smoothing='dv')
        nans = np.where(df['Mg'].isna().values, 1, 0).sum()
        no_nans = np.where(df['Mg'].isna().values, 0, 1).sum()
        self.assertEqual(no_nans, 137)
        self.assertEqual(nans, 7670)
        dkf = ld.get_Mg(tstart='2018-01-01', tend='2019-01-03', smoothing='kf')
        ddv = ld.get_Mg(tstart='2018-01-01', tend='2019-01-03', smoothing='dv')
        self.assertAlmostEqual(ddv.iloc[0]['Mg'], 344.0)
        self.assertAlmostEqual(dkf.iloc[0]['Mg'], 344.0)
        self.assertAlmostEqual(dkf.iloc[0]['Mg_err'], 50.0)

        dkf1 = ld.get_Mg(tstart='2018-11-01', tend='2019-01-03',
                         smoothing='kf')
        self.assertAlmostEqual(dkf1.iloc[0]['Mg'], 383.5)

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
        ld = LakeData()
        dkf = ld.get_ll(tstart=index[0], tend=index[-1], smoothing='kf')
        ddv = ld.get_ll(tstart=index[0], tend=index[-1], smoothing='dv')
        pd.testing.assert_frame_equal(kf_test_frame, dkf)
        pd.testing.assert_frame_equal(dv_test_frame, ddv)

    def test_lake_data_fits(self):
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
        warnings.filterwarnings("ignore", message="can't resolve package from")
        ld = LakeData()
        df = ld.get_data_fits('20160603', '20161231')
        tdf = pd.read_hdf(os.path.join(self.data_dir,
                                       'measurements_20160603_20161231.h5'),
                          'table')
        np.testing.assert_array_almost_equal(df['T'],
                                             tdf['T'], 1)
        np.testing.assert_array_almost_equal(df['z'],
                                             tdf['z'], 1)
        np.testing.assert_array_almost_equal(df['Mg'],
                                             tdf['Mg'], 0)

    # @unittest.skip("CSV reading needs to be fixed")
    def test_lake_data_csv(self):
        ti = self.load_input()
        with get_data('data/data.dat') as lb:
            ld = LakeData()
            vd = ld.get_data_csv(start='2003-01-16', end='2010-01-29', buf=lb)
            temp = [t for d, t in vd['T'].iteritems()]
            hgt = [h for d, h in vd['z'].iteritems()]
            mg = [m for d, m in vd['Mg'].iteritems()]
            cl = [c for d, c in vd['c'].iteritems()]
            o18 = [o for d, o in vd['o18'].iteritems()]
            h2 = [h for d, h in vd['h2'].iteritems()]
            dno = [d for dt, d in vd['nd'].iteritems()]
            dt = [dt for dt, d in vd['date'].iteritems()]
            np.testing.assert_array_almost_equal(temp, ti['temp'], 1)
            np.testing.assert_array_equal(dno, ti['nd'])
            np.testing.assert_array_almost_equal(hgt, ti['hgt'], 2)
            np.testing.assert_array_almost_equal(mg, ti['mg'], 3)
            np.testing.assert_array_almost_equal(cl, ti['cl'], 3)
            np.testing.assert_array_almost_equal(o18, ti['o18'], 2)
            np.testing.assert_array_almost_equal(h2, ti['h2'], 2)
            np.testing.assert_array_equal(
                np.array(dt, dtype='datetime64[ns]'), ti['date'])

        ld1 = LakeData()
        vd = ld1.get_data_csv(start='2003-01-16', end='2010-01-29')
        np.testing.assert_array_almost_equal(vd['T'], ti['temp'], 1)
        np.testing.assert_array_equal(vd['nd'], ti['nd'])
        np.testing.assert_array_almost_equal(vd['z'], ti['hgt'], 2)
        np.testing.assert_array_almost_equal(vd['Mg'], ti['mg'], 3)
        np.testing.assert_array_almost_equal(vd['c'], ti['cl'], 3)
        np.testing.assert_array_almost_equal(vd['o18'], ti['o18'], 2)
        np.testing.assert_array_almost_equal(vd['h2'], ti['h2'], 2)
        np.testing.assert_array_almost_equal(vd['o18'], ti['o18'], 2)
        np.testing.assert_array_almost_equal(vd['h2'], ti['h2'], 2)

    def test_wind_data_csv(self):
        ti = self.load_input()
        with get_data('data/wind.dat') as wb:
            dl = WindData(wb, default=0.0)
            df = dl.get_data(start='2003-01-16', end='2010-01-29')
            ws = [w for d, w in df.iteritems()]
            np.testing.assert_array_almost_equal(ws, ti['wind'], 1)

        dl1 = WindData(get_data('data/wind.dat'), default=0.0)
        df1 = dl1.get_data(start='2003-01-16', end='2010-01-29')
        ws = [w for d, w in df1.iteritems()]
        np.testing.assert_array_almost_equal(ws, ti['wind'], 1)


def suite():
    return unittest.makeSuite(DataTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
