import unittest
from unittest.mock import patch

import xarray as xr

from pumahu.uks import UnscentedKalmanSmoother
from pumahu.syn_model import (SynModel,
                              setup_test,
                              setup_realistic,
                              resample,
                              make_sparse)
from pumahu.visualise import trellis_plot


class VisTestCase(unittest.TestCase):

    def test_trellis_plot(self):
        xds4 = SynModel().run(setup_realistic(sinterval=120), addnoise=True)
        na = resample(xds4)
        na = make_sparse(na, ['m_out', 'X'])
        fig = trellis_plot(xds4, xr.Dataset({'exp': na}))
        self.assertEqual(len(fig['data']), 65)


if __name__ == '__main__':
    unittest.main()
