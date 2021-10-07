from collections import defaultdict
from datetime import datetime
import inspect
import os
import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from pumahu import get_data
from pumahu.data import LakeData, WindData


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
                               780.0)
        self.assertAlmostEqual(ld.FITS_request('z').iloc[0]['obs'],
                               2503.0)
        self.assertAlmostEqual(ld.FITS_request('T').iloc[0]['obs'],
                               24.4)

    def test_get_T(self):
        """
        Test different ways of receiving and smoothing temperature readings.
        """
        index = pd.date_range('2019-01-01', '2019-01-03')
        kf_test_frame = pd.DataFrame({'T': np.array([31.93833333,
                                                     31.84910086,
                                                     31.78383591]),
                                      'T_err': np.array([0.3, 0.08860859,
                                                         0.24334404]),
                                      'T_orig': np.array([31.93833333,
                                                          31.82958333,
                                                          31.98681818])},
                                     index=index)
        dv_test_frame = pd.DataFrame({'T': np.array([31.93833333,
                                                     31.82958333,
                                                     31.98681818]),
                                      'T_err': np.array([0.35497295,
                                                         0.66342641,
                                                         0.3821915])},
                                     index=index)
        ld = LakeData()
        dkf = ld.get_T(tstart='2019-01-01', tend='2019-01-03', smoothing='kf')
        ddv = ld.get_T(tstart='2019-01-01', tend='2019-01-03', smoothing='dv')
        pd.testing.assert_frame_equal(dv_test_frame, ddv)
        pd.testing.assert_frame_equal(kf_test_frame, dkf)

    def test_get_Mg(self):
        """
        Test different ways of receiving and smoothing temperature readings.
        """
        ld = LakeData()
        df = ld.get_Mg(tend='2019-07-31', smoothing='dv')
        nans = np.where(df['Mg'].isna().values, 1, 0).sum()
        no_nans = np.where(df['Mg'].isna().values, 0, 1).sum()
        self.assertEqual(no_nans, 573)
        self.assertEqual(nans, 19599)
        dkf = ld.get_Mg(tstart='2018-01-01', tend='2019-01-03',
                        smoothing='kf')
        ddv = ld.get_Mg(tstart='2018-01-01', tend='2019-01-03',
                        smoothing='dv')
        self.assertAlmostEqual(ddv.iloc[0]['Mg'], 344.0)
        self.assertAlmostEqual(dkf.iloc[0]['Mg'], 344.0)
        self.assertAlmostEqual(dkf.iloc[0]['Mg_err'], 50.0)

        dkf1 = ld.get_Mg(tstart='2018-11-01', tend='2019-01-03',
                         smoothing='kf')
        self.assertAlmostEqual(dkf1.iloc[0]['Mg'], 383.5)

    def test_get_ll(self):
        index = pd.date_range('2019-01-01', '2019-01-03')
        kf_test_frame = pd.DataFrame({'z': np.array([2529.32191667,
                                                     2529.33634745,
                                                     2529.33609614]),
                                      'z_err': np.array([0.03,
                                                         0.013484,
                                                         0.01401298]),
                                      'z_orig': np.array([2529.32191667,
                                                          2529.343125,
                                                          2529.33509091])},
                                     index=index)
        dv_test_frame = pd.DataFrame({'z': np.array([2529.32191667,
                                                     2529.343125,
                                                     2529.33509091]),
                                      'z_err': np.array([0.00672385,
                                                         0.01236163,
                                                         0.01548243])},
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
        df = ld.get_data('20160603', '20161231')
        tdf = xr.open_dataset(os.path.join(self.data_dir,
                               'measurements_20160603_20161231.nc'))
        np.testing.assert_array_almost_equal(df.loc[:, 'T', 'val'],
                                             tdf['T'], 1)
        np.testing.assert_array_almost_equal(df.loc[:, 'z', 'val'],
                                             tdf['z'], 1)
        np.testing.assert_array_almost_equal(df.loc[:, 'Mg', 'val'],
                                             tdf['Mg'], 0)

    def test_lake_data_csv(self):
        ld = LakeData(csvfile=get_data('data/data.csv'))
        xdf = ld.get_data('2000-1-1', '2021-1-1', ignore_cache=True)
        print(xdf)

    def test_wind_data_csv(self):
        ti = self.load_input()
        with open(get_data('data/wind.dat')) as wb:
            dl = WindData(wb, default=0.0)
            df = dl.get_data(start='2003-01-16', end='2010-01-29')
            ws = [w for d, w in df.iteritems()]
            np.testing.assert_array_almost_equal(ws, ti['wind'], 1)

        dl1 = WindData(get_data('data/wind.dat'), default=0.0)
        df1 = dl1.get_data(start='2003-01-16', end='2010-01-29')
        ws = [w for d, w in df1.iteritems()]
        np.testing.assert_array_almost_equal(ws, ti['wind'], 1)

    def test_outflow(self):
        """
        Test retrieving and interpolating outflow data.
        """
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
        ld = LakeData()
        ld.get_data('20190101', '20191231')
        ld.get_outflow()
        mean_mout = ld.xdf.loc[:, 'm_out', :].mean(axis=0).data 
        np.testing.assert_array_almost_equal(mean_mout,
                                             np.array([18.201107,
                                                       9.100553]), 6)

    def test_metservice_wind(self):
        """
        Test retrieving wind data from MetService wind model.
        """
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
        ld = LakeData()
        ld.get_data('20191201', '20191231')
        ld.get_MetService_wind()
        mean_ws = ld.xdf.loc[:, 'W', :].mean(axis=0).data 
        np.testing.assert_array_almost_equal(mean_ws,
                                             np.array([6.942369,
                                                       1.649878]), 6)
                                             

    def test_historic_data(self):
        """
        Test getting data for times when data collection was not
        done continuously. 
        """
        startdate = datetime(1994, 1, 1)
        enddate = datetime(1995, 12, 31)
        ld = LakeData()
        xdf = ld.get_data(startdate, enddate,
                          smoothing={'Mg': 2.6, 'T': 0.4, 'z': 0.01})
        nTobs = 30
        nZobs = 9
        nMgobs = 27
        self.assertEqual(nTobs, (~np.isnan(xdf.loc[:, 'T', 'val'].values)).sum()) 
        self.assertEqual(nZobs, (~np.isnan(xdf.loc[:, 'z', 'val'].values)).sum()) 
        self.assertEqual(nMgobs, (~np.isnan(xdf.loc[:, 'Mg', 'val'].values)).sum()) 
 

def suite():
    return unittest.makeSuite(DataTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
