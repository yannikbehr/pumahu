import unittest

import numpy as np

from pumahu import get_data
from pumahu.fd import fd 
from pumahu.syn_model import SynModel, setup_test
from pumahu.data import LakeData


class FDTestCase(unittest.TestCase):

    def test_synthetic_euler(self):
        sm = SynModel(integration_method='euler')
        data = sm.run(setup_test(), ignore_cache=True)
        rs = fd(data.exp, results_file=None, new=False, use_drmg=False,
                level2volume=sm.synth_fullness)
        q_test = rs.input.loc[:, 'q_in', 'val'].values
        q_exp = rs.exp.loc[:, 'q_in', 'val'].values
        q_diff = (q_test[:-1] - q_exp[:-1])
        rmse = np.sqrt(((q_diff) ** 2).mean())
        self.assertAlmostEqual(rmse, 17.37, 2)

    def test_historic_data(self):
        """
        Test with dataset used by Hurst et al. [2015]. 
        """
        ld = LakeData(csvfile=get_data('data/data.csv'))
        xdf = ld.get_data('2000-1-1', '2021-1-1',
                          smoothing={'Mg': 2.6, 'T': 0.4, 'z': 0.01})
        xdf = xdf.dropna('dates', how='all')
        rs = fd(xdf, results_file=None, new=False, use_drmg=False)
        rmse = np.sqrt(np.nanmean(rs.exp.loc[:, 'q_in', 'val'].values ** 2))
        self.assertAlmostEqual(rmse, 149.6, 2)


if __name__ == '__main__':
    unittest.main()