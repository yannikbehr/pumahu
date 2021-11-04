import unittest

import numpy as np

from pumahu.fd import fd 
from pumahu.syn_model import SynModel, setup_test


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
        self.assertAlmostEqual(rmse, 15.64, 2)


if __name__ == '__main__':
    unittest.main()