import unittest

import numpy as np

from clemb.clemb import Variable, Uniform, Gauss


class VariableTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_base_class(self):
        """
        Ensure that the base class can'b be instantiated.
        """
        self.assertRaises(TypeError, Variable)

    def test_uniform(self):
        """
        Ensure the values returned are uniformly distributed or identical to
        the input in case no variation is desired.
        """
        din = np.random.random_sample(10)
        u = Uniform(din)
        dout = [x for x in u]
        np.testing.assert_array_almost_equal(din, dout)

        din = np.ones(10000)
        u1 = Uniform(din)
        u1.min = 1.0
        u1.max = 0.0
        dout = np.array([x for x in u1])
        self.assertTrue(np.all(dout >= 0.0))
        self.assertTrue(np.all(dout < 1.0))
        # E[dout]=0.5
        self.assertTrue(abs(dout.mean() - 0.5) < 0.1)

    def test_gauss(self):
        """
        Ensure the values returned are normally distributed or identical to
        the input in case no variation is desired.
        """
        din = np.random.random_sample(10)
        u = Gauss(din)
        dout = [x for x in u]
        np.testing.assert_array_almost_equal(din, dout)

        din = np.zeros(1000)
        u1 = Gauss(din)
        u1.std = 0.1
        dout = np.array([x for x in u1])
        # E[dout]=0.0
        self.assertTrue(abs(dout.mean()) < 0.01)
        # std[dout] = 0.1
        self.assertTrue(abs(np.std(dout, ddof=1) - u1.std) < 0.01)


def suite():
    return unittest.makeSuite(VariableTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')